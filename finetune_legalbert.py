import sys
import pandas as pd
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import evaluate
import re
from load_legalbert import legalbert

theme = sys.argv[1]
test_defendant = sys.argv[2]
model_name = "nlpaueb/legal-bert-base-uncased"
max_length = 512
window_size = 10
dataset_path = "./data/annotated_transcripts.csv"

torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Prepare dataset: {theme}, {test_defendant}")
full = pd.read_csv(dataset_path)

# --- Preparing sliding window data --- #
sliding_window_df = pd.DataFrame()

for defendant in full["defendant"].unique():
    print(defendant)
    d = full[full["defendant"] == defendant].copy()
    d["idx"] = d["paragraph_id"].apply(lambda x: x.split("-")[-1])

    paragraph_series = d["text"].shift(0)
    for w in range(1, window_size):
        paragraph_series = paragraph_series + " " + d["text"].shift(-1 * w)

    idx_series = d["idx"].shift(0)
    for w in range(1, window_size):
        idx_series = idx_series + "," + d["idx"].shift(-1 * w)

    label_series = d[theme].shift(0)
    for w in range(1, window_size):
        label_series = label_series + d[theme].shift(-1 * w)
    label_series = label_series.apply(lambda l: 1 if l >= 1 else 0)
    x = pd.DataFrame(
        {
            "paragraph": paragraph_series,
            "paragraph_idx": idx_series,
            "label": label_series,  # predict if the paragraph contains at least one POSITIVE sentence
        }
    )
    x["defendant"] = defendant
    x = x[x["paragraph"].notnull()]
    x["paragraph"] = x["paragraph"].apply(
        lambda p: re.sub("\s?[\d]+\s", " ", p).strip()
    )

    sliding_window_df = pd.concat((sliding_window_df, x))

# --- Prepare training and test dataframe --- #
train_df = sliding_window_df[sliding_window_df["defendant"] != test_defendant].copy()
test_df = sliding_window_df[sliding_window_df["defendant"] == test_defendant].copy()

n_positives = len(train_df[train_df["label"] == 1])

# undersampling
train_df = pd.concat(
    (
        train_df[train_df["label"] == 1],
        (train_df[train_df["label"] == 0]).sample(n=3 * n_positives),
    )
)
train_df = train_df.sample(frac=1)
train_set = Dataset.from_pandas(train_df)

# --- Load tokenizer and model --- #
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
).to(device)


# set up finetuning dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_index, val_index = train_test_split(
    np.arange(len(train_df)), test_size=0.1, shuffle=True, random_state=527
)
val_df = train_df.iloc[val_index]
train_df = train_df.iloc[train_index]

X_train = train_df["paragraph"].to_list()
y_train = train_df["label"].astype(int).to_list()

X_val = val_df["paragraph"].to_list()
y_val = val_df["label"].astype(int).to_list()

train_encodings = tokenizer(
    X_train, truncation=True, padding=True, max_length=max_length
)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=max_length)

train_dataset = MyDataset(train_encodings, y_train)
eval_dataset = MyDataset(val_encodings, y_val)

# Set up Trainer and TrainingArguments for finetuning
training_args = TrainingArguments(
    output_dir=f"./results",  # output directory
    num_train_epochs=3,  # total number of training epochs
    optim="adamw_torch",
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,  # batch size for evaluation
    learning_rate=2e-5,  # initial learning rate for Adam optimizer
    warmup_steps=50,  # number of warmup steps for learning rate scheduler (set lower because of small dataset size)
    weight_decay=0.01,  # strength of weight decay
    evaluation_strategy="epoch",
    save_strategy="no",
)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Begin training model")
trainer.train()
print("Finished training model")

model_output_dir = Path(f"./models/legalbert-{theme}/")
model_output_dir.mkdir(exist_ok=True, parents=True)
trainer.save_model(model_output_dir / model_name.split("/")[-1])

print("Make prediction on test cases...")

trained_model = legalbert(model_output_dir / model_name.split("/")[-1])

corpus = test_df["paragraph"].to_list()
scores = trained_model.score(corpus)

test_output = list()
for i, item in enumerate(test_df.to_dict(orient="records")):
    input_ids = tokenizer(
        item["paragraph"], truncation=True, return_tensors="pt"
    ).input_ids
    test_output.append(
        {
            "paragraph": item["paragraph"],
            "score": scores[i],
            "label": item["label"],
            "paragraph_idx": item["paragraph_idx"],
            "len_tokens": len(input_ids[0]),
        }
    )

json.dump(
    test_output,
    open(
        f"./predictions/{model_name.split('/')[-1]}_sliding_{theme}_{test_defendant.replace(' ', '_')}.json",
        "w",
    ),
)
