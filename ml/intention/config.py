class ClassifierConfig:
    """Configuration class for model training and prediction parameters."""

    def __init__(
        self,
        model_name: str = "DeepPavlov/distilrubert-base-cased-conversational",
        dataset_path: str = "ml/rag/data/global_dataset.csv",
        save_path: str = "ml/models/intent-classifier",
        label_encoder_name: str = "label_encoder.pkl",
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 8,
        epochs: int = 3,
        weight_decay: float = 0.01,
        do_eval: bool = False,
        test_size: float = 0.2,
        output_hidden_states: bool = True,
        evaluation_strategy: str = "epoch",
        logging_dir: str = "./logs",
        logging_steps: int = 5000,
        save_strategy: str = "no",
        no_cuda: bool = False,
        threshold: float = 0.5,
        method: str = "words",
    ) -> None:
        # Training parameters
        self.MODEL_NAME = model_name
        self.DATASET_PATH = dataset_path
        self.SAVE_PATH = save_path
        self.LABEL_ENCODER_NAME = label_encoder_name
        self.DEVICE = device
        self.MAX_LENGTH = max_length
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.WEIGHT_DECAY = weight_decay
        self.DO_EVAL = do_eval
        self.TEST_SIZE = test_size

        # Model parameters
        self.OUTPUT_HIDDEN_STATES = output_hidden_states

        # Training strategy
        self.EVALUATION_STRATEGY = evaluation_strategy
        self.LOGGING_DIR = logging_dir
        self.LOGGING_STEPS = logging_steps
        self.SAVE_STRATEGY = save_strategy
        self.NO_CUDA = no_cuda

        # Prediction parameters
        self.THRESHOLD = threshold
        self.METHOD = method
