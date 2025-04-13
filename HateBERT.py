import os
import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import pathlib
from datetime import datetime

# Load the HateBERT model for hate speech detection
def load_hate_bert_model():
    """Load HateBERT model for hate speech detection."""
    print("Loading HateBERT model...")
    
    try:
        model_name = "GroNLP/hateBERT"
        model = pipeline("text-classification", model=model_name)
        print(f"Loaded {model_name}")
        return model
    except Exception as e:
        print(f"Failed to load HateBERT model: {e}")
        raise RuntimeError("Could not load HateBERT model for hate speech analysis")
            
def compute_hate_speech_scores(df, model, batch_size=16, max_length=512):
    """
    Compute hate speech scores for each statement using HateBERT.
    
    Args:
        df: DataFrame with 'statement' column
        model: Loaded HateBERT pipeline
        batch_size: Batch size for model processing
        max_length: Maximum length of text to process
        
    Returns:
        DataFrame with added 'label' and 'score' columns
    """
    result_df = df.copy()
    
    # Prepare lists for results
    all_labels = []
    all_scores = []
    
    # Use tqdm for a progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Computing hate speech scores"):
        batch = df["statement"].iloc[i:i+batch_size].tolist()
        try:
            batch_results = model(batch, truncation=True, max_length=max_length)
            for result in batch_results:
                all_labels.append(result["label"])
                all_scores.append(result["score"])
        except Exception as e:
            print(f"Error processing batch {i} to {i+batch_size}: {e}")
            # Fill with placeholder values for failed batch
            all_labels.extend(["ERROR"] * len(batch))
            all_scores.extend([np.nan] * len(batch))
    
    # Ensure results length matches DataFrame
    if len(all_labels) == len(df) and len(all_scores) == len(df):
        result_df["label"] = all_labels
        result_df["score"] = all_scores
    else:
        print(f"Warning: Results length mismatch. Expected {len(df)}, got {len(all_labels)} labels and {len(all_scores)} scores")
        # Pad with placeholders if necessary
        result_df["label"] = all_labels + ["ERROR"] * (len(df) - len(all_labels))
        result_df["score"] = all_scores + [np.nan] * (len(df) - len(all_scores))
    
    return result_df

def aggregate_hate_speech_by_year_party(df):
    """
    Aggregate hate speech classifications by year and party.
    
    Args:
        df: DataFrame with 'year', 'party', 'label', and 'score' columns
        
    Returns:
        DataFrame with aggregated scores
    """
    # Filter to include only republican, democrat, and moderator
    valid_parties = ['republican', 'democrat', 'moderator']
    filtered_df = df[df['party'].isin(valid_parties)].copy()
    
    # Convert label to binary (HateBERT uses 'hate' label for hate speech)
    filtered_df['is_hate_speech'] = filtered_df['label'].apply(lambda x: 1 if x.lower() == 'hate' else 0)
    
    # Group by year and party
    agg_df = filtered_df.groupby(['year', 'party']).agg(
        mean_score=('score', 'mean'),
        std_score=('score', 'std'),
        hate_speech_ratio=('is_hate_speech', 'mean'),
        count=('label', 'count')
    ).reset_index()
    
    # Replace NaN in std with 0 where count is 1
    agg_df.loc[agg_df['count'] == 1, 'std_score'] = 0
    
    return agg_df

def main(statements_csv_path, folder_name=None, output_base_dir='./output'):
    """
    Main function to run the analysis pipeline with HateBERT model.
    
    Args:
        statements_csv_path: Path to the CSV file with debate statements
        folder_name: Custom name for the output folder. If None, generates a name based on the input file.
        output_base_dir: Base directory for all outputs
    """
    # Generate output directory name if not provided
    if folder_name is None:
        # Get the input filename without extension
        input_file = pathlib.Path(statements_csv_path).stem
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{input_file}_HateBERT_{timestamp}"
    
    # Create complete output directory path
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Load the statements data
    print(f"Loading statements from {statements_csv_path}")
    df = pd.read_csv(statements_csv_path)
    print(f"Loaded {len(df)} statements")
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    # Party distribution
    print("\nParty distribution:")
    print(df['party'].value_counts())
    
    # Load the HateBERT model
    model = load_hate_bert_model()
    
    # Compute hate speech scores
    print("\nComputing hate speech scores...")
    df_with_scores = compute_hate_speech_scores(df, model)
    
    # Save detailed results
    detailed_output_path = os.path.join(output_dir, 'statements_with_hate_speech.csv')
    df_with_scores.to_csv(detailed_output_path, index=False)
    print(f"Saved detailed results to {detailed_output_path}")
    
    # Perform aggregation
    print("\nAggregating results by year and party...")
    agg_df = aggregate_hate_speech_by_year_party(df_with_scores)
    
    # Save aggregated results
    agg_output_path = os.path.join(output_dir, 'hate_speech_by_year_party.csv')
    agg_df.to_csv(agg_output_path, index=False)
    print(f"Saved aggregated results to {agg_output_path}")
    
    print(f"\nAnalysis complete! Data saved to {output_dir}")
    
    return output_dir  # Return the output directory path for reference

if __name__ == "__main__":
    # Examples of how to call the main function:
    # 1. With automatic folder name generation:
    # main("debate_statements_with_party.csv")
    
    # 2. With custom folder name:
    # main("debate_statements_with_party.csv", folder_name="HateBERT_Analysis")
    pass