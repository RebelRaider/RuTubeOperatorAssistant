import os
import pandas as pd
import json
from ml.constants import LOCAL_DATASETS_PATH, GLOBAL_DATASET_PATH, ALIASES_PATH

# Step 0: Create directories
os.makedirs(LOCAL_DATASETS_PATH, exist_ok=True)

# Step 1: Read CSV and keep required columns
df = pd.read_csv("ml/rag/dataset.csv")
db_df = pd.read_csv("ml/rag/db_dataset.csv")
df = df[["user_question", "label_1", "label_2"]]
db_df = db_df[["user_question", "label_1", "label_2"]]
df = pd.concat([df, db_df], ignore_index=True)
print(df)
# Step 2: Create global_dataset.csv
global_df = df[["user_question", "label_1"]]
global_df.to_csv(GLOBAL_DATASET_PATH, index=False)

# Step 3: Process groups and create aliases
aliases = {}
grouped = df.groupby("label_1")

for label_1, group in grouped:
    unique_label_2 = group["label_2"].unique()
    if len(unique_label_2) == 1:
        aliases[label_1] = unique_label_2[0]
    else:
        local_df = group[["user_question", "label_2"]]
        print(label_1, local_df["label_2"].unique())
        filename = f"{LOCAL_DATASETS_PATH}/{label_1}.csv"
        local_df.to_csv(filename, index=False)

# Step 4: Save aliases
with open(ALIASES_PATH, "w") as f:
    json.dump(aliases, f, ensure_ascii=False, indent=4)
