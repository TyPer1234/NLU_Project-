#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import pathlib
from datetime import datetime
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_hatexplain_model():
    """
    Load the HateXplain model and tokenizer, and move the model to the proper device.
    """
    print("Loading HateXplain model...")
    try:
        # Use the full model identifier.
        model_name = "HateXplain/roberta-base-hatexplain"
        # Set device (MPS if available, otherwise CPU)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # If the repository is gated, you need to pass your Hugging Face auth token.
        hx_tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        hx_model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=True).to(device)
        hx_model.eval()  # Set to evaluation mode.
        
        print(f"Loaded {model_name}")
        return hx_model, hx_tokenizer, device
    except Exception as e:
        print(f"Failed to load HateXplain model: {e}")
        raise RuntimeError("Could not load HateXplain model for hostility analysis.")

def compute_hate_speech_scores(df, hx_model, hx_tokenizer, device, batch_size=16, max_length=512):
    """
    Compute hate speech scores for each statement using HateXplain in batches.
    
    Args:
        df: DataFrame with a 'statement' column.
        hx_model: Loaded HateXplain model.
        hx_tokenizer: Loaded HateXplain tokenizer.
        device: Torch device for inference.
        batch_size: Number of statements per batch.
        max_length: Maximum tokenized sequence length.
        
    Returns:
        DataFrame with added 'label' and 'score' columns.
    """
    result_df = df.copy()
    all_labels = []
    all_scores = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Computing hate speech scores"):
        batch_statements = df["statement"].iloc[i:i+batch_size].tolist()
        
        encodings = hx_tokenizer(
            batch_statements,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        encodings = {key: tensor.to(device) for key, tensor in encodings.items()}
        
        with torch.no_grad():
            outputs = hx_model(**encodings)
            probs = softmax(outputs.logits, dim=1)
        
        labels = probs.argmax(dim=1)
        scores = probs.max(dim=1).values
        
        for label_id, score in zip(labels, scores):
            label_name = hx_model.config.id2label[label_id.item()]
            all_labels.append(label_name)
            all_scores.append(score.item())
    
    result_df["label"] = all_labels
    result_df["score"] = all_scores
    return result_df

def aggregate_hate_speech_by_year_party(df):
    """
    Aggregate hate speech classifications by year and party.
    
    Args:
        df: DataFrame containing 'year', 'party', 'label', and 'score' columns.
        
    Returns:
        Aggregated DataFrame with statistics per group.
    """
    valid_parties = ['republican', 'democrat', 'moderator']
    filtered_df = df[df['party'].isin(valid_parties)].copy()
    
    # Assume that the label "Hateful" (case-insensitive) indicates hate speech.
    filtered_df['is_hate_speech'] = filtered_df['label'].apply(lambda x: 1 if x.lower() == 'hateful' else 0)
    
    agg_df = filtered_df.groupby(['year', 'party']).agg(
        mean_score=('score', 'mean'),
        std_score=('score', 'std'),
        hate_speech_ratio=('is_hate_speech', 'mean'),
        count=('label', 'count')
    ).reset_index()
    
    agg_df.loc[agg_df['count'] == 1, 'std_score'] = 0
    return agg_df

def main(statements_csv_path, folder_name=None, output_base_dir='./output'):
    """
    Run the HateXplain analysis pipeline.
    
    Args:
        statements_csv_path: Path to a CSV file with at least the columns 'statement', 'year', and 'party'.
        folder_name: Custom folder name for saving outputs.
        output_base_dir: Base directory for outputs.
    
    Outputs:
        - 'statements_with_hate_speech.csv': Detailed per-statement predictions.
        - 'hate_speech_by_year_party.csv': Aggregated hate speech analysis.
    """
    if folder_name is None:
        input_file = pathlib.Path(statements_csv_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{input_file}_HateXplain_{timestamp}"
    
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    print(f"Loading statements from {statements_csv_path}...")
    df = pd.read_csv(statements_csv_path)
    print(f"Loaded {len(df)} statements")
    
    print("\nSample data:")
    print(df.head())
    
    print("\nParty distribution:")
    print(df['party'].value_counts())
    
    hx_model, hx_tokenizer, device = load_hatexplain_model()
    
    print("\nComputing hate speech scores using HateXplain...")
    df_with_scores = compute_hate_speech_scores(df, hx_model, hx_tokenizer, device)
    
    detailed_output_path = os.path.join(output_dir, 'statements_with_hate_speech.csv')
    df_with_scores.to_csv(detailed_output_path, index=False)
    print(f"Saved detailed results to {detailed_output_path}")
    
    print("\nAggregating hate speech results by year and party...")
    agg_df = aggregate_hate_speech_by_year_party(df_with_scores)
    agg_output_path = os.path.join(output_dir, 'hate_speech_by_year_party.csv')
    agg_df.to_csv(agg_output_path, index=False)
    print(f"Saved aggregated results to {agg_output_path}")
    
    print("\nAnalysis complete!")
    return output_dir

if __name__ == "__main__":
    # Example usage:
    # main("debate_statements_with_party_by_statement.csv",
    #      folder_name="HateXplain_Analysis_by_statement", 
    #      output_base_dir="./output/HateXplain")
    pass