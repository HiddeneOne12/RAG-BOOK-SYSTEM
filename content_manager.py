import os
import logging
from typing import List, Dict, Any
from rag_system import RAGSystem
from website_processor import WebsiteProcessor

logger = logging.getLogger(__name__)

class ContentManager:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.website_processor = WebsiteProcessor()
        self.processed_sources = set()
    
    def add_pdf(self, pdf_path: str) -> bool:
        """Add a PDF file to the knowledge base"""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            documents = self.rag_system.process_pdf(pdf_path)
            self.rag_system.add_documents_to_collection(documents, os.path.basename(pdf_path))
            self.processed_sources.add(pdf_path)
            logger.info(f"Successfully added PDF: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding PDF {pdf_path}: {str(e)}")
            return False
    
    def add_website(self, url: str) -> bool:
        """Add a website to the knowledge base"""
        try:
            # Clean and validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            logger.info(f"Attempting to process website: {url}")
            
            # Try to validate URL first
            if not self.website_processor.validate_url(url):
                logger.error(f"Invalid or inaccessible URL: {url}")
                return False
            
            # Process the website
            documents = self.website_processor.process_website(url)
            
            if not documents or len(documents) == 0:
                logger.error(f"No content extracted from {url}")
                return False
            
            # Add to knowledge base
            self.rag_system.add_documents_to_collection(documents, url)
            self.processed_sources.add(url)
            logger.info(f"Successfully added website: {url} ({len(documents)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding website {url}: {str(e)}")
            return False
    
    def process_pdf_folder(self, folder_path: str) -> Dict[str, Any]:
        """Process all PDFs in a folder"""
        try:
            if not os.path.exists(folder_path):
                return {"success": False, "message": f"Folder not found: {folder_path}"}
            
            pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
            
            if not pdf_files:
                return {"success": False, "message": "No PDF files found in folder"}
            
            results = {"success": True, "processed": [], "failed": []}
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)
                if self.add_pdf(pdf_path):
                    results["processed"].append(pdf_file)
                else:
                    results["failed"].append(pdf_file)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF folder: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer"""
        try:
            return self.rag_system.ask_question(question)
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'context_documents': []
            }
    
    def get_processed_sources(self) -> List[str]:
        """Get list of processed sources"""
        return list(self.processed_sources)
    
    def clear_knowledge_base(self) -> bool:
        """Clear the knowledge base"""
        try:
            # This would require reinitializing the ChromaDB collection
            # For now, we'll just clear the processed sources list
            self.processed_sources.clear()
            logger.info("Knowledge base cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return False
