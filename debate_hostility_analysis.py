import os
import pandas as pd
import numpy as np
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm
from datetime import datetime
import pathlib

# Load the sentiment analysis model for hostility scoring
def load_toxicity_model():
    """Load a pre-trained model for toxicity/hostility detection."""
    print("Loading toxicity/hostility model...")   
    try:
        model_name = "unitary/unbiased-toxic-roberta"
        toxicity_pipeline = pipeline("text-classification", model=model_name)
        print(f"Loaded {model_name}")
        return toxicity_pipeline
    except Exception as e:
        print(f"Failed to load Unitary model: {e}")
        raise RuntimeError("Could not load Unitary model for hostility analysis")
            
def compute_hostility_scores(df, model, batch_size=16):
    """
    Compute hostility scores for each statement in the dataframe.
    
    Args:
        df: DataFrame with 'statement' column
        model: Loaded huggingface pipeline for sentiment/toxicity
        batch_size: Batch size for model processing
        
    Returns:
        DataFrame with added 'hostility_score' column
    """
    result_df = df.copy()
    
    # Process in batches to avoid memory issues
    all_scores = []
    
    # Use tqdm for a progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Computing hostility scores"):
        batch = df["statement"].iloc[i:i+batch_size].tolist()
        try:
            batch_scores = model(batch, truncation=True, max_length=512)
            all_scores.extend([result["score"] for result in batch_scores])
        except Exception as e:
            print(f"Error processing batch {i} to {i+batch_size}: {e}")
            # Fill with NaN values for failed batch
            all_scores.extend([np.nan] * len(batch))
    
    # Ensure scores length matches DataFrame
    if len(all_scores) == len(df):
        result_df["hostility_score"] = all_scores
    else:
        print(f"Warning: Score length mismatch. Expected {len(df)}, got {len(all_scores)}")
        # Pad with NaN if necessary
        result_df["hostility_score"] = all_scores + [np.nan] * (len(df) - len(all_scores))
    
    return result_df

def aggregate_hostility_by_year_party(df):
    """
    Aggregate hostility scores by year and party.
    
    Args:
        df: DataFrame with 'year', 'party', and 'hostility_score' columns
        
    Returns:
        DataFrame with aggregated scores
    """
    # Filter to include only republican, democrat, and moderator
    valid_parties = ['republican', 'democrat', 'moderator']
    filtered_df = df[df['party'].isin(valid_parties)].copy()
    
    # Group by year and party
    agg_df = filtered_df.groupby(['year', 'party'])['hostility_score'].agg(
        mean_hostility=('mean'),
        std_hostility=('std'),
        count=('count')
    ).reset_index()
    
    # Replace NaN in std with 0 where count is 1
    agg_df.loc[agg_df['count'] == 1, 'std_hostility'] = 0
    
    return agg_df

def main(statements_csv_path, folder_name=None, output_base_dir='./output'):
    """
    Main function to run the analysis pipeline without plotting.
    
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
        folder_name = f"{input_file}_{timestamp}"
    
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
    
    # Load the toxicity model
    model = load_toxicity_model()
    
    # Compute hostility scores
    print("\nComputing hostility scores...")
    df_with_scores = compute_hostility_scores(df, model)
    
    # Save detailed results
    detailed_output_path = os.path.join(output_dir, 'statements_with_hostility.csv')
    df_with_scores.to_csv(detailed_output_path, index=False)
    print(f"Saved detailed results to {detailed_output_path}")
    
    # Aggregate by year and party
    print("\nAggregating results by year and party...")
    agg_df = aggregate_hostility_by_year_party(df_with_scores)
    
    # Save aggregated results
    agg_output_path = os.path.join(output_dir, 'hostility_by_year_party.csv')
    agg_df.to_csv(agg_output_path, index=False)
    print(f"Saved aggregated results to {agg_output_path}")
    
    print(f"\nAnalysis complete! Data saved to {output_dir}")
    
    return output_dir  # Return the output directory path for reference

if __name__ == "__main__":
    # Examples of how to call the main function:
    # 1. With automatic folder name generation:
    # main("debate_statements_with_party.csv")
    
    # 2. With custom folder name:
    # main("debate_statements_with_party.csv", folder_name="2020_debates_analysis")
    pass