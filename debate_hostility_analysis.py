import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Load the sentiment analysis model for hostility scoring
# You can choose different models based on what you have access to
def load_toxicity_model():
    """Load a pre-trained model for toxicity/hostility detection."""
    print("Loading toxicity/hostility model...")
    
    # Option 1: RoBERTa-based toxicity model
    try:
        model_name = "unitary/unbiased-toxic-roberta"
        toxicity_pipeline = pipeline("text-classification", model=model_name)
        print(f"Loaded {model_name}")
        return toxicity_pipeline
    except Exception as e:
        print(f"Failed to load primary model: {e}")
        
        # Option 2: Fallback to sentiment analysis
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
            print(f"Loaded fallback model: {model_name}")
            
            # Create a wrapper function to convert sentiment to toxicity scores
            # (where negative sentiment is treated as higher hostility)
            def sentiment_to_hostility(texts, **kwargs):
                results = sentiment_pipeline(texts, **kwargs)
                # Convert sentiment to hostility score (POSITIVE → low hostility, NEGATIVE → high hostility)
                hostility_scores = []
                for result in results:
                    if result['label'] == 'NEGATIVE':
                        hostility_scores.append({'score': result['score']})
                    else:
                        hostility_scores.append({'score': 1 - result['score']})
                return hostility_scores
            
            return sentiment_to_hostility
        except Exception as e:
            print(f"Failed to load fallback model: {e}")
            raise RuntimeError("Could not load any suitable model for hostility analysis")

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

def plot_hostility_trends(agg_df, output_path='hostility_trends.png'):
    """
    Create a line plot showing hostility trends over time by party.
    
    Args:
        agg_df: Aggregated DataFrame from aggregate_hostility_by_year_party
        output_path: Path to save the plot
    """
    # Set figure size and style
    plt.figure(figsize=(14, 8))
    
    # Define colors and labels for each party
    party_colors = {
        'republican': 'red',
        'democrat': 'blue',
        'moderator': 'gray'
    }
    
    party_labels = {
        'republican': 'Republican',
        'democrat': 'Democrat',
        'moderator': 'Moderator'
    }
    
    # Get unique years for x-axis
    years = sorted(agg_df['year'].unique())
    
    # Plot each party's trend
    for party in ['democrat', 'republican', 'moderator']:
        party_data = agg_df[agg_df['party'] == party]
        
        if len(party_data) > 0:
            plt.plot(
                party_data['year'], 
                party_data['mean_hostility'],
                marker='o',
                markersize=8,
                linewidth=2.5,
                color=party_colors[party],
                label=party_labels[party]
            )
            
            # Add error bands (standard deviation)
            plt.fill_between(
                party_data['year'],
                party_data['mean_hostility'] - party_data['std_hostility'],
                party_data['mean_hostility'] + party_data['std_hostility'],
                color=party_colors[party],
                alpha=0.2
            )
    
    # Add a trendline for overall hostility
    overall_by_year = agg_df.groupby('year')['mean_hostility'].mean().reset_index()
    plt.plot(
        overall_by_year['year'],
        overall_by_year['mean_hostility'],
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Overall Trend'
    )
    
    # Customize the plot
    plt.title('Hostility in Presidential Debates (1960-2024)', fontsize=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Hostility Score', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    
    # Set x-axis to show every 4 years (election years)
    min_year = min(years)
    max_year = max(years)
    plt.xticks(range(min_year, max_year + 1, 4), rotation=45)
    
    # Adjust y-axis to start from 0
    plt.ylim(0, plt.ylim()[1])
    
    # Add annotations for key historical events/context
    # (Optional, can be uncommented and customized)
    """
    annotations = {
        1980: "First Reagan-Carter Debate",
        2000: "Bush-Gore",
        2016: "Trump-Clinton",
        2020: "COVID Era\nTrump-Biden"
    }
    
    for year, text in annotations.items():
        if year in years:
            y_pos = agg_df[agg_df['year'] == year]['mean_hostility'].mean()
            plt.annotate(text, xy=(year, y_pos), xytext=(year-1, y_pos+0.1),
                         arrowprops=dict(arrowstyle='->'), fontsize=12)
    """
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show plot
    plt.show()

def main(statements_csv_path, output_dir='./output'):
    """
    Main function to run the complete analysis pipeline.
    
    Args:
        statements_csv_path: Path to the CSV file with debate statements
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    detailed_output_path = os.path.join(output_dir, 'debate_statements_with_hostility.csv')
    df_with_scores.to_csv(detailed_output_path, index=False)
    print(f"Saved detailed results to {detailed_output_path}")
    
    # Aggregate by year and party
    print("\nAggregating results by year and party...")
    agg_df = aggregate_hostility_by_year_party(df_with_scores)
    
    # Save aggregated results
    agg_output_path = os.path.join(output_dir, 'hostility_by_year_party.csv')
    agg_df.to_csv(agg_output_path, index=False)
    print(f"Saved aggregated results to {agg_output_path}")
    
    # Plot the results
    print("\nGenerating visualization...")
    plot_path = os.path.join(output_dir, 'hostility_trends.png')
    plot_hostility_trends(agg_df, plot_path)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # You can call the main function with your CSV path
    # Example: main("debate_statements_with_party.csv")
    pass