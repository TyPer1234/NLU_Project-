# Presidential Debate Hostility Analysis

This repository contains a pipeline for scraping, processing, and analyzing presidential debate transcripts to compute hostility scores by year and party. The goal is to track trends in hostility and aggressive rhetoric over time in U.S. presidential debates.

## Goal

The goal of this project is to analyze hostility levels in U.S. presidential debates over time by:

1. Scraping Transcripts: Retrieving debate transcripts from the UCSB Presidency Project.
2. Processing Text: Identifying speakers, their party affiliation, and their statements. Text is split by sentence or by paragraph depending on the parameter provided.
3. Calculating Hostility Scores: Using a pretrained model to compute hostility scores for each statement.
4. Visualizing Trends: Aggregating scores by year and party to generate visualizations of hostility trends over time.

## Scripts Breakdown

### debate_scraper.py
This script fetches debate transcripts from the UCSB website.

- Retrieves debates from a list of URLs and saves them to the `data_script/` directory.
- Outputs text files named according to their debate date and type:
  - `pb`: Presidential debates
  - `vp`: Vice presidential debates
  - `rpd`: Republican debates
  - `dcd`: Democratic debates

### debate_processing.py
Processes debate transcripts by identifying speakers and their statements.

- Extracts statements by splitting by sentence or paragraph (controlled via a parameter `split_by_sentence`).
- Identifies speakers and their political affiliations (Republican, Democrat, Moderator) using a dictionary of known speakers.
- Saves processed results in `.csv` files:
  - `debate_statements_with_party_by_sentence.csv` (if split by sentence)
  - `debate_statements_with_party_by_statement.csv` (if split by paragraph)

#### Text Processing

This is a simplified sentence tokenizer that ensures common abbreviations are not split mistakenly:
```python
def simple_sentence_tokenize(text):
    if not text:
        return []
        
    text = re.sub(r'(Mr\.|Mrs\.|Ms\.|Dr\.|Gov\.|Sen\.)\s', r'\1PLACEHOLDER', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [re.sub(r'PLACEHOLDER', ' ', s) for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


