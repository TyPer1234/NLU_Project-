import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime

# Constants
URL = "https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0"
SAVE_DIR = "data_script"
PRESIDENTIAL_DIR = os.path.join(SAVE_DIR, "presidential")

# Create output directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRESIDENTIAL_DIR, exist_ok=True)

def fetch_debate_links():
    """Fetch debate transcript links from the UCSB website."""
    try:
        response = requests.get(URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return {}

    link_dict = {}

    for link in soup.find_all('tr'):
        date_td = link.find('td', style=lambda x: x and "width:112pt" in x)
        if not date_td:
            continue

        debate_date = date_td.get_text(strip=True)
        if len(debate_date) > 30:
            continue

        name_td = date_td.find_next("td")
        if not name_td:
            continue

        debate_name = name_td.get_text(strip=True)
        if any(word in debate_name.lower() for word in ['cancelled']):
            continue

        link_tag = name_td.find("a")
        hyperlink = link_tag.get("href", None)
        if not hyperlink:
            continue

        link_dict[debate_name] = [debate_date, hyperlink]

    return link_dict

def format_filename(debate_name, debate_date):
    """Format filenames for saving transcripts with year_month_day format."""
    try:
        date_obj = datetime.strptime(debate_date, "%B %d, %Y")
        formatted_date = date_obj.strftime("%Y_%m_%d")  # Year-Month-Day format
    except ValueError:
        print(f"Skipping invalid date: {debate_date}")
        return None  

    # Categorization logic
    if "Vice" in debate_name:
        suffix = "vp"
    elif debate_name.startswith("Presidential"):
        suffix = "pd"
    elif "Republican" in debate_name:
        suffix = "rpd"
    elif "Democratic" in debate_name:
        suffix = "dcd"
    else:
        suffix = "unknown"

    filename = f"{formatted_date}_{suffix}.txt"
    return re.sub(r'[<>:"/\\|?*]', '', filename), suffix  # Sanitize filename

def scrape_and_save_transcripts(link_dict):
    """Scrape debate transcripts and save them as text files."""
    for title, debate_info in link_dict.items():
        date, url = debate_info

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
        except requests.RequestException as e:
            print(f"Error fetching transcript: {e}")
            continue

        content_div = soup.find("div", class_="field-docs-content")
        if not content_div:
            continue

        paragraphs = content_div.find_all("p")
        text_content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])

        filename, suffix = format_filename(title, date)
        if not filename:
            continue  

        # Save in presidential/ if it's a general election debate
        folder_path = PRESIDENTIAL_DIR if suffix == "pd" else SAVE_DIR
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_content)
        
        print(f"Saved: {file_path}")

def main():
    """Main function to execute the script."""
    print("Fetching debate links...")
    debate_links = fetch_debate_links()

    if not debate_links:
        print("No debate links found. Exiting...")
        return

    print(f"Found {len(debate_links)} debates. Scraping transcripts...")
    scrape_and_save_transcripts(debate_links)
    print("All transcripts saved successfully.")

if __name__ == "__main__":
    main()