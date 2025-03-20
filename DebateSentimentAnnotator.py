#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import random
import json
import re
from datetime import datetime

class DebateSentimentAnnotator:
    def __init__(self, data_dir='data', output_file='debate_sentiment_ratings.json'):
        """Initialize the debate sentiment annotator tool."""
        self.data_dir = data_dir
        self.output_file = output_file
        self.ratings = self._load_existing_ratings()
        self.statements = []
        self.current_file = ""
    
    def _load_existing_ratings(self):
        """Load existing ratings from file if it exists."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading ratings file. Starting with empty ratings.")
                return {}
        return {}
    
    def _save_ratings(self):
        """Save current ratings to file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.ratings, f, indent=2)
        print(f"Ratings saved to {self.output_file}")
    
    def _get_debate_files(self):
        """Get list of all debate transcript files."""
        return glob.glob(os.path.join(self.data_dir, "*.txt"))
    
    def _parse_filename(self, filename):
        """Parse the filename to get debate details."""
        basename = os.path.basename(filename)
        parts = basename.split('_')
        
        date_parts = parts[:3]
        date_str = f"{date_parts[0]}/{date_parts[1]}/{date_parts[2]}"
        
        debate_type = ""
        if parts[3].startswith("vp"):
            debate_type = "Vice Presidential Debate"
        elif parts[3].startswith("rpd"):
            debate_type = "Republican Primary Debate"
        elif parts[3].startswith("dcd"):
            debate_type = "Democratic Primary Debate"
        elif parts[3].startswith("pb"):
            debate_type = "Presidential Debate"
        
        return {
            "date": date_str,
            "type": debate_type,
            "filename": basename
        }
    
    def _extract_statements(self, file_path):
        """Extract individual statements from debate transcript."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the content into speaker segments
        # This pattern looks for uppercase speaker names followed by colon
        # or Mr./Mrs./President/Senator/Governor/Vice followed by a name and colon
        pattern = r'([A-Z][A-Z\s\.]+:|(?:Mr\.|Mrs\.|President|Senator|Governor|Vice)\s+[A-Za-z]+:)'
        segments = re.split(pattern, content)
        
        statements = []
        speaker = ""
        
        for i, segment in enumerate(segments):
            if i % 2 == 0 and i > 0:  # This is content
                # Clean up the statement
                text = segment.strip()
                if text and len(text) > 20:  # Only keep substantial statements
                    statements.append({
                        "speaker": speaker.strip(':'),
                        "statement": text,
                        "file_info": self._parse_filename(file_path),
                        "statement_id": f"{os.path.basename(file_path)}-{i//2}"
                    })
            else:  # This is a speaker
                speaker = segment
        
        return statements
    
    def load_debates(self):
        """Load all debate transcripts and extract statements."""
        files = self._get_debate_files()
        all_statements = []
        
        for file_path in files:
            try:
                statements = self._extract_statements(file_path)
                all_statements.extend(statements)
                print(f"Loaded {len(statements)} statements from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        self.statements = all_statements
        print(f"Total statements loaded: {len(self.statements)}")
    
    def get_random_statement(self):
        """Get a random statement that hasn't been rated yet."""
        unrated_statements = [s for s in self.statements 
                              if s["statement_id"] not in self.ratings]
        
        if not unrated_statements:
            print("All statements have been rated! Starting over with all statements.")
            return random.choice(self.statements)
        
        return random.choice(unrated_statements)
    
    def start_annotation(self):
        """Start the interactive annotation process."""
        if not self.statements:
            print("No statements loaded. Please run load_debates() first.")
            return
        
        print("\n=== Presidential Debate Sentiment Annotation Tool ===")
        print("Rate statements on hostility:")
        print("  2 = Quite hostile")
        print("  1 = Sort of hostile")
        print("  0 = Not hostile")
        print("Enter 'q' to quit and save, 's' to save and continue\n")
        
        try:
            while True:
                statement = self.get_random_statement()
                print("\n" + "="*80)
                print(f"Debate: {statement['file_info']['type']} ({statement['file_info']['date']})")
                print(f"Speaker: {statement['speaker']}")
                print("-"*80)
                print(f"Statement: {statement['statement']}")
                print("-"*80)
                
                rating = input("Hostility rating (0-2), 'q' to quit, 's' to save, 'skip' to skip: ")
                
                if rating.lower() == 'q':
                    break
                elif rating.lower() == 's':
                    self._save_ratings()
                    continue
                elif rating.lower() == 'skip':
                    continue
                
                try:
                    rating_value = int(rating)
                    if rating_value not in [0, 1, 2]:
                        print("Invalid rating. Please enter 0, 1, or 2.")
                        continue
                    
                    # Save the rating
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.ratings[statement["statement_id"]] = {
                        "rating": rating_value,
                        "statement": statement["statement"],
                        "speaker": statement["speaker"],
                        "debate_info": statement["file_info"],
                        "timestamp": timestamp
                    }
                    
                    print(f"Rated as: {rating_value}")
                except ValueError:
                    print("Invalid input. Please enter 0, 1, 2, 'q', 's', or 'skip'.")
        
        finally:
            self._save_ratings()
            print(f"\nAnnotation session ended. {len(self.ratings)} statements rated.")
    
    def get_statistics(self):
        """Print basic statistics about the current ratings."""
        if not self.ratings:
            print("No ratings available.")
            return
        
        total = len(self.ratings)
        rating_counts = {0: 0, 1: 0, 2: 0}
        
        for statement_id, data in self.ratings.items():
            rating = data["rating"]
            rating_counts[rating] += 1
        
        print("\n=== Sentiment Rating Statistics ===")
        print(f"Total statements rated: {total}")
        print(f"Not hostile (0): {rating_counts[0]} ({(rating_counts[0]/total)*100:.1f}%)")
        print(f"Sort of hostile (1): {rating_counts[1]} ({(rating_counts[1]/total)*100:.1f}%)")
        print(f"Quite hostile (2): {rating_counts[2]} ({(rating_counts[2]/total)*100:.1f}%)")

