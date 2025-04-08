# Presidential Debate Hostility Analysis

This repository aims to analyze how general sentiment and hostility in U.S. presidential debates have evolved over time. The analysis is divided into three major focuses:

1. **Sentiment and Hostility Analysis:** Utilizing fine-tuning techniques on Large Language Models (LLMs), particularly BERT-based models, to detect hostile language within presidential debates.
2. **Temporal Analysis:** Examining how sentiment and hostility fluctuate over time, identifying trends and shifts in rhetorical tone across election cycles.
3. **Topic-Specific Sentiment:** Identifying specific political topics that are more likely to involve hostile rhetoric and exploring the relationship between sentiment intensity and issue-based discussions.

## Scripts Breakdown

The analysis pipeline is structured around three main Python scripts:

### 1. debate_scraper.py (Scraping)
This script retrieves U.S. presidential debate transcripts from the UCSB Presidency Project website.

- **Objective:** Download all available transcripts for general election debates.
- **Output:** Raw text files saved in the `data_script/` directory.
- **Format:** Files are named according to their debate date and type:
  - `pb`: Presidential debates
  - `vp`: Vice presidential debates
  - `rpd`: Republican debates
  - `dcd`: Democratic debates

### 2. debate_processing.py (Processing)
Processes downloaded debate transcripts by identifying speakers, their party affiliation, and their statements.

- **Objective:** Structure the raw debate text into identifiable statements by speakers, categorized by party.
- **Approach:** 
  - Uses a dictionary of known speakers for each debate.
  - Applies sentence tokenization or paragraph processing based on the specified parameter.
  - Identifies speakers’ political affiliation (Republican, Democrat, Moderator) using a pre-defined mapping.
- **Output:** 
  - `debate_statements_with_party_by_sentence.csv`: Processed data with statements split by sentence.
  - `debate_statements_with_party_by_statement.csv`: Processed data with statements split by paragraph.

### 3. debate_hostility_analysis.py (Hostility Analysis)
Performs the main analysis by computing hostility scores using pretrained NLP models.

- **Objective:** Compute hostility scores for each statement, aggregate them by year and party, and visualize trends over time.
- **Approach:** 
  - Loads a pretrained BERT-based model to compute hostility scores.
  - Aggregates hostility scores by year and party.
  - Plots the results to identify trends over election cycles.
- **Output:** 
  - `debate_hostility_scores.csv`: Contains aggregated hostility scores by year and party.
  - Visualizations of hostility trends over time.
