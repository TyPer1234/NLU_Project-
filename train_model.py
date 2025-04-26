import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification, BertConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from transformers import EarlyStoppingCallback

# Load function 
def load_and_prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing labels
    df = df[~df['hostility_label'].isna()]
    df['hostility_label'] = df['hostility_label'].astype(int)
    
    # Rename hostility column
    df = df.rename(columns={'hostility_label': 'labels'})
    
    # Keep only 'statement' and 'labels'
    df = df[['statement', 'labels']]
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    return dataset

# tokenizer
def tokenize_function(data, tokenizer):
    return tokenizer(data["statement"], truncation=True, padding="max_length",max_length=512)

# metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer on a Hostility Classification Task")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the Hugging Face model to use")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the CSV file containing the dataset")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save the model and results")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    # Load and preprocess dataset
    dataset = load_and_prepare_dataset(args.csv_path)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Train-validation split
    dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        disable_tqdm=False, # progress bar
        use_mps_device=torch.backends.mps.is_available()
    )
    
    print(torch.backends.mps.is_available())
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Executing trainer
    trainer.train()