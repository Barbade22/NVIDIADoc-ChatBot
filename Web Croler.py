import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

class CudaDocsCrawler:
    def __init__(self, base_url, max_depth, max_pages):
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.data = []

    def crawl(self):
        self.crawl_page(self.base_url, depth=1)

        # Save the collected data to a file
        self.save_data()

    def crawl_page(self, url, depth):
        if depth > self.max_depth or len(self.data) >= self.max_pages or url in self.visited_urls:
            return
        
        self.visited_urls.add(url)

        # Fetch the web page content
        response = requests.get(url)
        if response.status_code == 200:
            # Parse HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract plain text and clean it
            text = self.extract_text(soup)
            
            # Save the text along with URL
            self.data.append({
                'url': url,
                'content': text
            })

            # Extract links and recursively crawl them
            links = soup.find_all('a', href=True)
            for link in links:
                next_url = urljoin(self.base_url, link['href'])
                if next_url.startswith(self.base_url) and next_url not in self.visited_urls:
                    self.crawl_page(next_url, depth + 1)

    def extract_text(self, soup):
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        
        # Extract text and clean it
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())
        return text

    def save_data(self):
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"Scraped data saved to 'cuda_docs_crawled_data.json'.")

# Example usage:
if __name__ == "__main__":
    base_url = 'https://docs.nvidia.com/cuda/'
    max_depth = 5
    max_pages = 10
    
    crawler = CudaDocsCrawler(base_url, max_depth, max_pages)
    crawler.crawl()
