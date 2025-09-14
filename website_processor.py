import requests
import re
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class WebsiteProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_content(self, url: str) -> str:
        """Extract clean text content from a website"""
        try:
            # Set headers to mimic a real browser more effectively
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Add session for better handling
            session = requests.Session()
            session.headers.update(headers)
            
            # Try to get the page
            response = session.get(url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                element.decompose()
            
            # For Wikipedia, try to get the main content
            if 'wikipedia.org' in url:
                main_content = soup.find('div', {'id': 'mw-content-text'})
                if main_content:
                    # Remove navigation and other Wikipedia-specific elements
                    for element in main_content.find_all(['div'], class_=['navbox', 'infobox', 'thumb', 'reference']):
                        element.decompose()
                    text = main_content.get_text()
                else:
                    text = soup.get_text()
            else:
                text = soup.get_text()
            
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Ensure we have meaningful content
            if len(text) < 100:
                raise ValueError("Insufficient content extracted")
            
            logger.info(f"Extracted {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            raise
    
    def process_website(self, url: str) -> List[Document]:
        """Process website and return chunked documents"""
        try:
            # Extract content
            content = self.extract_content(url)
            
            # Create document
            doc = Document(
                page_content=content, 
                metadata={
                    "source": url, 
                    "type": "website",
                    "title": self._extract_title(url, content)
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Processed {len(chunks)} chunks from {url}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing website {url}: {str(e)}")
            raise
    
    def _extract_title(self, url: str, content: str) -> str:
        """Extract title from content or use URL"""
        try:
            # Try to find title in content
            title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
            
            # Use domain name as fallback
            domain = url.split('//')[-1].split('/')[0]
            return domain
            
        except:
            return url
    
    def validate_url(self, url: str) -> bool:
        """Validate if URL is accessible"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Use the same headers as extract_content for consistency
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            # Try HEAD request first, fallback to GET if HEAD fails
            try:
                response = session.head(url, timeout=10, allow_redirects=True)
                return response.status_code == 200
            except:
                # Some sites don't support HEAD, try GET
                response = session.get(url, timeout=10, allow_redirects=True)
                return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {str(e)}")
            return False
