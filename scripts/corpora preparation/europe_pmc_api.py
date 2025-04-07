import argparse
import os
import re
from time import sleep

import requests

DEFAULT_KEYWORDS_FILE = "datasets/keywords/default_keywords.txt"
DEFAULT_OUTPUT_DIR = "outputs/Europe PMC API"
DEFAULT_MAX_DOWNLOADS = 2831

class EuropePMCFetcher:
    def __init__(self, output_dir, keywords, max_downloads):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.keywords = [k.lower() for k in keywords]
        self.max_downloads = max_downloads

        self.search_query = self.build_search_query(self.keywords)

        self.params = {
            "query": self.search_query,
            "resultType": "core",
            "pageSize": 1000,
            "format": "json",
            "cursorMark": "*"
        }

    def build_search_query(self, keywords):
        return " OR ".join(f'"{kw}"' if " " in kw else kw for kw in keywords)

    def sanitize_filename(self, filename):
        return re.sub(r'[<>:"/\\|?*]', "_", filename)

    def download_pdf(self, pdf_url):
        try:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                filename = self.sanitize_filename(pdf_url.split("/")[-1].split("?")[0] + ".pdf")
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")
        return False

    def fetch_and_download(self):
        print(f"Search query: {self.search_query}")
        print("Fetching initial result count from Europe PMC...")
        initial_params = self.params.copy()
        initial_params["pageSize"] = 1
        response = requests.get(self.base_url, params=initial_params)
        response.raise_for_status()
        total_results = int(response.json().get("hitCount", 0))

        print(f"Total results available: {total_results}")
        if total_results == 0:
            print("No results found. Exiting.")
            return

        print(f"Starting download (max {self.max_downloads} articles)...")

        total_downloaded = 0
        total_processed = 0

        while total_downloaded < self.max_downloads:
            try:
                response = requests.get(self.base_url, params=self.params)
                response.raise_for_status()
                data = response.json()

                articles = data.get("resultList", {}).get("result", [])
                next_cursor = data.get("nextCursorMark")

                if not articles:
                    break

                for article in articles:
                    metadata = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
                    if not any(k in metadata for k in self.keywords):
                        continue

                    links = article.get("fullTextUrlList", {}).get("fullTextUrl", [])
                    pdf_url = next((l["url"] for l in links if l.get("documentStyle") == "pdf"), None)

                    if pdf_url and self.download_pdf(pdf_url):
                        total_downloaded += 1
                    total_processed += 1

                    if total_downloaded >= self.max_downloads:
                        break

                self.params["cursorMark"] = next_cursor
                sleep(1)

            except Exception as e:
                print(f"Error while processing: {e}")
                continue

        print("\nDownload Summary:")
        print(f"Total articles processed: {total_processed}")
        print(f"Total PDFs downloaded: {total_downloaded}")

def load_keywords(keywords_file):
    try:
        with open(keywords_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Could not load keywords from {keywords_file}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Fetch and download PDF articles from Europe PMC using one keyword list.")
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f"Folder to save the returned PDF articles (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('--keywords_file', default=DEFAULT_KEYWORDS_FILE,
                        help=f"Text file containing search queries, with one query per line (default: {DEFAULT_KEYWORDS_FILE})")
    parser.add_argument('--max_downloads', type=int, default=DEFAULT_MAX_DOWNLOADS,
                        help=f"Maximum number of PDF articles to download (default: {DEFAULT_MAX_DOWNLOADS})")

    args = parser.parse_args()

    keywords = load_keywords(args.keywords_file)
    fetcher = EuropePMCFetcher(
        output_dir=args.output_dir,
        keywords=keywords,
        max_downloads=args.max_downloads
    )
    fetcher.fetch_and_download()

if __name__ == "__main__":
    main()
