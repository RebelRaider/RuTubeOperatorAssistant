from typing import Dict, Optional, Tuple
import os
import joblib
import pandas as pd
import torch
from datasets import Dataset
from ml.intention.config import ClassifierConfig  # Adjusted import
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


class TrainingHandler:
    """Class for training and evaluating transformer-based models."""

    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        dataset_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        if not config:
            config = ClassifierConfig()
        self.config = config
        self.model_name = config.MODEL_NAME
        self.dataset_path = dataset_path if dataset_path else config.DATASET_PATH
        self.save_path = save_path if save_path else config.SAVE_PATH
        self.label_encoder_name = config.LABEL_ENCODER_NAME
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.NO_CUDA else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.label_encoder = LabelEncoder()
        self.max_length = config.MAX_LENGTH
        self.batch_size = config.BATCH_SIZE
        self.epochs = config.EPOCHS
        self.weight_decay = config.WEIGHT_DECAY
        self.do_eval = config.DO_EVAL
        self.test_size = config.TEST_SIZE

    def _tokenize_data(self, examples):
        """Tokenize the data."""
        return self.tokenizer(
            examples["user_question"],
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def _create_dataset(self, df: pd.DataFrame) -> Dataset:
        """Create a HuggingFace Dataset from a DataFrame."""
        dataset = Dataset.from_pandas(df)
        # Apply label encoding
        dataset = dataset.map(
            lambda x: {"labels": self.label_encoder.transform([x["label"]])[0]}
        )
        # Tokenize the dataset
        dataset = dataset.map(self._tokenize_data, batched=True)
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    def load_and_prepare_data(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load and prepare data for training and evaluation."""
        df = pd.read_csv(self.dataset_path)
        # Ensure the label column is named 'label'
        if "label_1" in df.columns:
            df.rename(columns={"label_1": "label"}, inplace=True)
        elif "label_2" in df.columns:
            df.rename(columns={"label_2": "label"}, inplace=True)
        else:
            raise ValueError("Label column not found in the dataset.")

        # Fit label encoder
        self.label_encoder.fit(df["label"])
        # Save label encoder
        joblib.dump(
            self.label_encoder, os.path.join(self.save_path, self.label_encoder_name)
        )

        if self.do_eval:
            train_df, eval_df = train_test_split(
                df, test_size=self.test_size, stratify=df["label"], random_state=42
            )
            train_dataset = self._create_dataset(train_df)
            eval_dataset = self._create_dataset(eval_df)
            return train_dataset, eval_dataset
        else:
            train_dataset = self._create_dataset(df)
            return train_dataset, None

    def _compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        preds = (
            p.predictions[0].argmax(-1)
            if isinstance(p.predictions, tuple)
            else p.predictions.argmax(-1)
        )
        labels = p.label_ids
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train_model(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> AutoModelForSequenceClassification:
        """Train the model."""
        os.makedirs(self.save_path, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.save_path,
            overwrite_output_dir=True,
            eval_strategy=self.config.EVALUATION_STRATEGY if self.do_eval else "no",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            logging_dir=self.config.LOGGING_DIR,
            logging_steps=self.config.LOGGING_STEPS,
            save_strategy=self.config.SAVE_STRATEGY,
            no_cuda=self.config.NO_CUDA,
            seed=42,
            load_best_model_at_end=self.do_eval,
            metric_for_best_model="f1",
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label_encoder.classes_)
        ).to(self.device)

        data_collator = DataCollatorWithPadding(self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if self.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics if self.do_eval else None,
        )

        trainer.train()
        self.trainer = trainer
        return model

    def save_model_and_tokenizer(
        self, model: AutoModelForSequenceClassification
    ) -> None:
        """Save the model and tokenizer."""
        model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def _evaluate_model(self):
        """Evaluate the model."""
        eval_metrics = self.trainer.evaluate()
        print("Evaluation Metrics:")
        print(f"Accuracy: {eval_metrics['eval_accuracy']}")
        print(f"Precision: {eval_metrics['eval_precision']}")
        print(f"Recall: {eval_metrics['eval_recall']}")
        print(f"F1-Score: {eval_metrics['eval_f1']}")

    def run(self) -> None:
        """Run the training and evaluation process."""
        train_dataset, eval_dataset = self.load_and_prepare_data()
        model = self.train_model(train_dataset, eval_dataset)
        self.save_model_and_tokenizer(model)

        if self.do_eval:
            self._evaluate_model()


if __name__ == "__main__":
    # Step 1: Create necessary directories
    os.makedirs("trained_models/global-model", exist_ok=True)

    # Step 2: Train and save the global model
    global_trainer = TrainingHandler(
        dataset_path="rag/global_dataset.csv",
        save_path="trained_models/global-model",
    )
    global_trainer.run()

    # Step 3: Iterate over local rag
    local_datasets_path = "rag/local-rag"
    trained_models_path = "trained_models"

    for csv_file in os.listdir(local_datasets_path):
        if csv_file.endswith(".csv"):
            dataset_name = os.path.splitext(csv_file)[0]
            dataset_path = os.path.join(local_datasets_path, csv_file)
            save_path = os.path.join(trained_models_path, dataset_name)
            os.makedirs(save_path, exist_ok=True)

            print(f"\nTraining model for dataset: {dataset_name}")

            local_trainer = TrainingHandler(
                dataset_path=dataset_path,
                save_path=save_path,
            )
            local_trainer.run()
