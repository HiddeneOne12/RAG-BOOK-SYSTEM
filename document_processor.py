"""
Document Processing Module
Handles PDF and website document processing and chunking
"""

import os
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and chunking operations"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process PDF file and return chunked documents
        
        **How PDF Processing Works:**
        1. **PyPDFLoader**: Uses LangChain's PyPDFLoader to extract text from PDF
        2. **Text Extraction**: Converts PDF pages to plain text
        3. **Chunking**: Splits text into smaller chunks using RecursiveCharacterTextSplitter
        4. **Metadata**: Adds source information and chunk IDs
        
        **Packages Used:**
        - `langchain_community.document_loaders.PyPDFLoader`: PDF text extraction
        - `langchain.text_splitter.RecursiveCharacterTextSplitter`: Text chunking
        - `langchain.schema.Document`: Document structure
        
        **Chunking Strategy:**
        - Chunk Size: 1000 characters (configurable)
        - Overlap: 200 characters between chunks
        - Separators: Paragraphs, lines, words, characters
        """
        try:
            # Step 1: Load PDF using PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Step 2: Split into chunks using RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Processed {len(chunks)} chunks from {pdf_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def process_website(self, url: str) -> List[Document]:
        """
        Process website content and return chunked documents
        
        **How Website Processing Works:**
        1. **HTTP Request**: Uses requests library to fetch webpage content
        2. **HTML Parsing**: Uses BeautifulSoup to extract text from HTML
        3. **Content Cleaning**: Removes HTML tags and normalizes whitespace
        4. **Chunking**: Splits content into manageable chunks
        5. **Metadata**: Adds source URL and type information
        
        **Packages Used:**
        - `requests`: HTTP requests to fetch web content
        - `beautifulsoup4`: HTML parsing and text extraction
        - `lxml`: XML/HTML parser backend
        - `langchain.text_splitter.RecursiveCharacterTextSplitter`: Text chunking
        
        **Processing Steps:**
        1. Send HTTP GET request to URL
        2. Parse HTML content with BeautifulSoup
        3. Extract text content, removing HTML tags
        4. Clean and normalize whitespace
        5. Create Document object with metadata
        6. Split into chunks for vector storage
        """
        try:
            # Step 1: Fetch webpage content
            import requests
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Step 2: Parse HTML and extract text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Step 3: Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Step 4: Get text content
            text = soup.get_text()
            
            # Step 5: Clean and normalize whitespace
            import re
            clean_content = re.sub(r'\s+', ' ', text).strip()
            
            # Step 6: Create document
            doc = Document(page_content=clean_content, metadata={"source": url, "type": "website"})
            
            # Step 7: Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Processed {len(chunks)} chunks from {url}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing website {url}: {str(e)}")
            raise
    
    def process_pdf_folder(self, folder_path: str) -> List[Document]:
        """Process all PDFs in a folder"""
        all_documents = []
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                documents = self.process_pdf(pdf_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        return all_documents
