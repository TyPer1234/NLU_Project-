from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, Dataset
import pandas as pd
import torch

def load_holdout(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['statement', 'hostility_label']].rename(columns={'hostility_label': 'labels'})
    df = df[~df['labels'].isna()]
    return Dataset.from_pandas(df)

def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["statement"], padding="max_length", truncation=True), batched=True)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Load model + tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./output_test")
tokenizer = AutoTokenizer.from_pretrained("./output_test")

# Load and prepare your held-out test set
dataset = load_holdout("PrimaryVPHostileLabels.csv")
tokenized = tokenize_data(dataset, tokenizer)

# Optional: filter down to a clean test split
eval_dataset = tokenized.train_test_split(test_size=0.2, seed=1)["test"]

# Run evaluation
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate(eval_dataset=eval_dataset)
print("Evaluation Results:", results)