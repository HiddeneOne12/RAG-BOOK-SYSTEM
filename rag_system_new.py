"""
RAG System - Main Module
Simplified RAG system with modular architecture
"""

import os
import logging
from typing import List, Dict, Any
from langchain.schema import Document

from config import PDF_FOLDER, OPENROUTER_API_KEY, OPENROUTER_MODEL
from language_processor import LanguageProcessor
from ai_response_generator import AIResponseGenerator
from document_processor import DocumentProcessor
from vector_database import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Main RAG System Class
    
    **Architecture Overview:**
    - LanguageProcessor: Handles language detection and translation
    - DocumentProcessor: Processes PDFs and websites
    - VectorDatabase: Manages ChromaDB operations
    - AIResponseGenerator: Generates AI responses using various models
    
    **How the Complete System Works:**
    
    1. **Document Processing:**
       - PDF: PyPDFLoader → Text extraction → Chunking → Vector storage
       - Website: requests + BeautifulSoup → HTML parsing → Text extraction → Chunking → Vector storage
    
    2. **Question Processing:**
       - Language detection (Arabic/English)
       - Query translation (if needed to match document language)
       - Vector similarity search
       - Context retrieval with relevance scoring
    
    3. **Answer Generation:**
       - RAG response generation (Ollama/API/Fallback)
       - OpenRouter enhancement (if API key available)
       - Language-specific response formatting
    
    4. **Vector Database:**
       - ChromaDB for persistent storage
       - SentenceTransformer for embeddings
       - Cosine similarity for document retrieval
    """
    
    def __init__(self):
        """Initialize RAG system components"""
        self.language_processor = LanguageProcessor()
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabase()
        self.ai_generator = AIResponseGenerator(self.language_processor)
        
        logger.info("RAG system components initialized successfully")
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF file and return chunked documents"""
        return self.document_processor.process_pdf(pdf_path)
    
    def process_website(self, url: str) -> List[Document]:
        """Process website content and return chunked documents"""
        return self.document_processor.process_website(url)
    
    def add_documents_to_collection(self, documents: List[Document], source_name: str):
        """Add documents to ChromaDB collection"""
        self.vector_db.add_documents(documents, source_name)
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents with intelligent language conversion"""
        try:
            # Convert query language to match document language if needed
            converted_query = self.language_processor.convert_query_language(query, self.vector_db.collection)
            return self.vector_db.search_similar_documents(converted_query, n_results)
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_documents: List[Dict]) -> str:
        """Generate answer using RAG first, then enhance with OpenRouter API"""
        try:
            if not context_documents:
                return "❌ **Not Found in Documents**\n\nI couldn't find relevant information in the uploaded PDFs or websites to answer your question. Please try rephrasing your question or add more relevant documents."
            
            # Check if we have good quality context
            relevant_docs = [doc for doc in context_documents if doc.get('relevance_score', 0) > 0.3]
            if not relevant_docs:
                return "❌ **Not Found in Documents**\n\nI found some documents but they don't contain relevant information to answer your question. Please try a different question or add more relevant documents."
            
            # Prepare context with relevance scoring
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(f"[Relevance: {doc.get('relevance_score', 0):.2f}] {doc['document']}")
            
            context = "\n\n".join(context_parts)
            
            # Step 1: Generate RAG response
            rag_response = self.ai_generator.generate_rag_response(query, context)

            # Step 2: Enhance with OpenRouter
            if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
                try:
                    enhanced_response = self.ai_generator.enhance_with_openrouter(query, rag_response, context)
                    return enhanced_response
                except Exception as e:
                    logger.error(f"OpenRouter enhancement failed: {str(e)}")
                    return rag_response
            else:
                return rag_response
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"❌ **Error**\n\nI encountered an error while generating the answer. Please try rephrasing your question."
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Main method to ask a question and get an answer"""
        try:
            similar_docs = self.search_similar_documents(question, n_results=3)
            answer = self.generate_answer(question, similar_docs)
            
            return {
                'question': question,
                'answer': answer,
                'sources': [doc['metadata']['source'] for doc in similar_docs],
                'context_documents': similar_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'context_documents': []
            }
    
    def process_pdf_folder(self):
        """Process all PDFs in the PDF folder"""
        try:
            pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {PDF_FOLDER}")
                return
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(PDF_FOLDER, pdf_file)
                logger.info(f"Processing {pdf_file}...")
                
                documents = self.process_pdf(pdf_path)
                self.add_documents_to_collection(documents, pdf_file)
                logger.info(f"Successfully processed {pdf_file}")
                
        except Exception as e:
            logger.error(f"Error processing PDF folder: {str(e)}")
            raise
