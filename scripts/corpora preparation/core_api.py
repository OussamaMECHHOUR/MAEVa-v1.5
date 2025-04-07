import argparse
import os

import requests

DEFAULT_KEYWORDS_FILE = "datasets/keywords/default_keywords.txt"
DEFAULT_OUTPUT_DIR = "outputs/Core API"

class CoreAPIFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.core.ac.uk/v3/search/works/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def search_articles(self, query, limit=5000):
        params = {"q": query, "scroll": True, "limit": limit}
        response = requests.get(self.base_url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def download_articles(self, json_response, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        articles = json_response.get('results', [])

        filtered = [
            {
                'title': a['title'],
                'downloadUrl': a.get('downloadUrl')
            }
            for a in articles
            if a.get('documentType') == 'research' and a.get('language', {}).get('code') == 'en'
        ]

        print(f"Number of articles with download URLs: {sum(1 for a in filtered if a['downloadUrl'])}")

        for article in filtered:
            title, url = article['title'], article['downloadUrl']
            if url:
                try:
                    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", ".", "_", "-")).strip()
                    file_path = os.path.join(folder_path, f"{safe_title}.pdf")

                    r = requests.get(url, stream=True)
                    r.raise_for_status()

                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"Downloaded: {safe_title}")
                except Exception as e:
                    print(f"Failed to download {title}: {e}")

def load_keywords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Could not read keyword file {file_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Fetch and download articles from CORE API.")
    parser.add_argument('--api_key', required=True, help="CORE API key")
    parser.add_argument('--query_file', default=DEFAULT_KEYWORDS_FILE,
                        help=f"Text file containing search queries, with one query per line (default: {DEFAULT_KEYWORDS_FILE})")
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f"Folder to save the returned PDF articles (default: {DEFAULT_OUTPUT_DIR})")

    args = parser.parse_args()

    fetcher = CoreAPIFetcher(api_key=args.api_key)
    keywords = load_keywords(args.query_file)

    for query in keywords:
        print(f"\nProcessing query: {query}")
        try:
            results = fetcher.search_articles(query=query)
            query_folder = os.path.join(args.output_dir, query.replace(" ", "_"))
            fetcher.download_articles(results, folder_path=query_folder)
        except Exception as e:
            print(f"Error while processing query '{query}': {e}")
            break

if __name__ == "__main__":
    main()

