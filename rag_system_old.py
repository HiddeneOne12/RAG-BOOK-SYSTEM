import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import requests
import json

from config import (
    CHROMA_PERSIST_DIRECTORY, 
    PDF_FOLDER, 
    MODEL_NAME, 
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_TIMEOUT,
    API_FALLBACK_ENABLED,
    API_FALLBACK_URL,
    API_FALLBACK_MODEL,
    API_FALLBACK_TIMEOUT,
    API_KEY
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.tokenizer = None
        self.model = None
        self.text_splitter = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG system components"""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("RAG system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    def load_model(self):
        """Load the model with quantization for lower memory usage"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                logger.info("Model loaded with quantization")
                
            except Exception as quant_error:
                logger.warning(f"Quantization failed, loading without: {quant_error}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,
                    device_map="auto"
                )
                logger.info("Model loaded without quantization")
            
            logger.info(f"Model {MODEL_NAME} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF file and return chunked documents"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Processed {len(chunks)} chunks from {pdf_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def process_website(self, url: str) -> List[Document]:
        """Process website content and return chunked documents"""
        try:
            # Simple web scraping
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract text content (basic implementation)
            content = response.text
            # Remove HTML tags (basic)
            import re
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            # Create document
            doc = Document(page_content=clean_content, metadata={"source": url, "type": "website"})
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Processed {len(chunks)} chunks from {url}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing website {url}: {str(e)}")
            raise
    
    def add_documents_to_collection(self, documents: List[Document], source_name: str):
        """Add documents to ChromaDB collection"""
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [{"source": source_name, "chunk_id": i, "type": documents[0].metadata.get("type", "pdf")} for i in range(len(documents))]
            ids = [f"{source_name}_{i}" for i in range(len(documents))]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to collection: {str(e)}")
            raise
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents in the collection with maximum accuracy"""
        try:
            # Convert query language to match document language if needed
            converted_query = self._convert_query_language(query)
            query_embedding = self.embedding_model.encode([converted_query]).tolist()[0]
            
            # Search with more results for better context
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    # Only include results with good similarity (distance < 0.8)
                    if results['distances'][0][i] < 0.8:
                      formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i],
                            'relevance_score': 1 - results['distances'][0][i]  # Higher score = more relevant
                        })
            
            # Sort by relevance score (highest first)
            formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Return top 3 most relevant results
            return formatted_results[:3]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
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
            rag_response = self._generate_rag_response(query, context)

            # Step 2: Enhance with OpenRouter
            if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
                try:
                    enhanced_response = self._enhance_rag_with_openrouter(query, rag_response, context)
                    return enhanced_response
                except Exception as e:
                    logger.error(f"OpenRouter enhancement failed: {str(e)}")
                    return rag_response
            else:
                return rag_response
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"❌ **Error**\n\nI encountered an error while generating the answer. Please try rephrasing your question."
    
    def _generate_rag_response(self, query: str, context: str) -> str:
        """Generate RAG response using existing methods"""
        try:
            if self._check_ollama_available():
                try:
                    return self._generate_ollama_answer(query, context)
                except Exception as e:
                    logger.error(f"Error with Ollama: {str(e)}")
                    if API_FALLBACK_ENABLED and API_KEY:
                        return self._generate_api_answer(query, context)
                    else:
                        return self._fallback_answer(query, context)
            else:
                if API_FALLBACK_ENABLED and API_KEY:
                    return self._generate_api_answer(query, context)
                else:
                    return self._fallback_answer(query, context)
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return self._fallback_answer(query, context)
    
    def _generate_ollama_answer(self, query: str, context: str) -> str:
        """Generate answer using Ollama with precision focus"""
        try:
            # Check if user wants details
            wants_details = any(word in query.lower() for word in ['explain', 'detail', 'how', 'why', 'what is', 'describe'])
            
            # Detect if question is in Arabic
            is_arabic = self._is_arabic_text(query)
            
            system_prompt = f"""You are an expert research assistant. Provide precise, concise answers based on the provided content.

**Response Rules:**
- For simple questions: 2-3 lines maximum
- For detailed questions: Provide comprehensive explanation
- Only use information from the provided context
- If information is not available: Say "Not found in the provided documents"
- Be direct and accurate

**Language Support:**
- Respond ONLY in the same language as the question
- If question is in Arabic: Respond ONLY in Arabic
- If question is in English: Respond ONLY in English
- Do NOT provide translations"""
            
            user_prompt = f"""Question: {query}
Language: {'Arabic' if is_arabic else 'English'}

Context from documents:
{context}

Provide a precise answer based on the context above.
{'Respond ONLY in Arabic' if is_arabic else 'Respond ONLY in English'}."""
            
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.3, "max_tokens": 800 if wants_details else 400}
            }
            
            response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', 'No response generated')
            else:
                return self._fallback_answer(query, context)
                
        except Exception as e:
            logger.error(f"Error with Ollama: {str(e)}")
            return self._fallback_answer(query, context)
    
    def _enhance_rag_with_openrouter(self, query: str, rag_response: str, context: str) -> str:
        """Enhance RAG response using OpenRouter API for precision and conciseness"""
        try:
            system_prompt = """You are an expert research assistant. Your task is to enhance RAG responses to be:

1. **PRECISE**: Answer exactly what was asked, no more, no less
2. **CONCISE**: 2-3 lines for simple answers, detailed only when specifically requested
3. **ACCURATE**: Base everything on the provided context from documents
4. **CLEAR**: Use simple language and clear structure

**Response Rules:**
- For simple questions: 2-3 lines maximum
- For "explain in detail" questions: Provide comprehensive explanation
- If information is not in the context: Say "Not found in the provided documents"
- Use bullet points for lists
- Bold important terms
- Be direct and to the point

**Language Support:**
- Respond ONLY in the same language as the question
- If question is in Arabic: Respond ONLY in Arabic
- If question is in English: Respond ONLY in English
- Do NOT provide translations"""
            
            # Detect if question is in Arabic
            is_arabic = self._is_arabic_text(query)
            
            user_prompt = f"""Enhance this RAG response to be precise and concise:

**Question:** {query}
**Language:** {'Arabic' if is_arabic else 'English'}

**RAG Response to Enhance:**
{rag_response}

**Supporting Context from Documents:**
{context[:1000]}

**Instructions:**
- Make it precise and concise (2-3 lines unless details are requested)
- Only use information from the provided context
- If information is missing, say "Not found in the provided documents"
- Be direct and clear
- {'Respond ONLY in Arabic' if is_arabic else 'Respond ONLY in English'}"""
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.3,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=OPENROUTER_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if enhanced_content:
                    return f"{enhanced_content}\n\n---\n*🤖 Response enhanced using OpenRouter AI*"
                else:
                    return rag_response
            else:
                logger.error(f"OpenRouter API Error: {response.status_code}")
                return rag_response
                
        except Exception as e:
            logger.error(f"Error enhancing with OpenRouter: {str(e)}")
            return rag_response
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _is_arabic_text(self, text: str) -> bool:
        """Detect if text contains Arabic characters"""
        arabic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                # Check if character is in Arabic Unicode range
                if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\u08A0' <= char <= '\u08FF' or '\uFB50' <= char <= '\uFDFF' or '\uFE70' <= char <= '\uFEFF':
                    arabic_chars += 1
        
        # If more than 30% of alphabetic characters are Arabic, consider it Arabic text
        return total_chars > 0 and (arabic_chars / total_chars) > 0.3
    
    def _detect_document_language(self) -> str:
        """Detect the primary language of documents in the collection"""
        try:
            # Get a sample of documents from the collection
            results = self.collection.get(limit=10)
            if not results['documents'] or len(results['documents']) == 0:
                return "english"  # Default to English
            
            # Analyze sample documents
            arabic_count = 0
            total_docs = len(results['documents'])
            
            for doc in results['documents']:
                if self._is_arabic_text(doc):
                    arabic_count += 1
            
            # If more than 50% of documents are Arabic, consider collection Arabic
            if arabic_count / total_docs > 0.5:
                return "arabic"
            else:
                return "english"
                
        except Exception as e:
            logger.error(f"Error detecting document language: {str(e)}")
            return "english"  # Default to English
    
    def _convert_query_language(self, query: str) -> str:
        """Convert query language to match document language"""
        try:
            # Detect document language
            doc_language = self._detect_document_language()
            query_language = "arabic" if self._is_arabic_text(query) else "english"
            
            # If query is Arabic but documents are English, translate to English
            if query_language == "arabic" and doc_language == "english":
                return self._translate_arabic_to_english(query)
            # If query is English but documents are Arabic, translate to Arabic
            elif query_language == "english" and doc_language == "arabic":
                return self._translate_english_to_arabic(query)
            else:
                # Same language, return as is
                return query
                
        except Exception as e:
            logger.error(f"Error converting query language: {str(e)}")
            return query
    
    def _translate_arabic_to_english(self, arabic_text: str) -> str:
        """Translate Arabic text to English using OpenRouter API"""
        try:
            if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
                return arabic_text  # Return original if no API key
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a professional translator. Translate the given Arabic text to English accurately while preserving the meaning and context."
                    },
                    {
                        "role": "user", 
                        "content": f"Translate this Arabic text to English: {arabic_text}"
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.3,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                translation = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if translation:
                    logger.info(f"Translated Arabic query to English: {arabic_text} -> {translation}")
                    return translation.strip()
            
            return arabic_text
            
        except Exception as e:
            logger.error(f"Error translating Arabic to English: {str(e)}")
            return arabic_text
    
    def _translate_english_to_arabic(self, english_text: str) -> str:
        """Translate English text to Arabic using OpenRouter API"""
        try:
            if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
                return english_text  # Return original if no API key
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a professional translator. Translate the given English text to Arabic accurately while preserving the meaning and context."
                    },
                    {
                        "role": "user", 
                        "content": f"Translate this English text to Arabic: {english_text}"
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.3,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                translation = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if translation:
                    logger.info(f"Translated English query to Arabic: {english_text} -> {translation}")
                    return translation.strip()
            
            return english_text
            
        except Exception as e:
            logger.error(f"Error translating English to Arabic: {str(e)}")
            return english_text
    
    def _generate_api_answer(self, query: str, context: str) -> str:
        """Generate answer using API fallback with precision focus"""
        if not API_FALLBACK_ENABLED or not API_KEY:
            return self._fallback_answer(query, context)
    
        try:
            # Check if user wants details
            wants_details = any(word in query.lower() for word in ['explain', 'detail', 'how', 'why', 'what is', 'describe'])
            
            # Detect if question is in Arabic
            is_arabic = self._is_arabic_text(query)
            
            system_prompt = """You are an expert research assistant. Provide precise, concise answers based on the provided content.

**Response Rules:**
- For simple questions: 2-3 lines maximum
- For detailed questions: Provide comprehensive explanation
- Only use information from the provided context
- If information is not available: Say "Not found in the provided documents"
- Be direct and accurate

**Language Support:**
- Respond ONLY in the same language as the question
- If question is in Arabic: Respond ONLY in Arabic
- If question is in English: Respond ONLY in English
- Do NOT provide translations"""
            
            user_prompt = f"""Question: {query}
Language: {'Arabic' if is_arabic else 'English'}

Context from documents:
{context}

Provide a precise answer based on the context above.
{'Respond ONLY in Arabic' if is_arabic else 'Respond ONLY in English'}."""
            
            payload = {
                "model": API_FALLBACK_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 800 if wants_details else 400,
                "temperature": 0.3
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            
            response = requests.post(API_FALLBACK_URL, json=payload, headers=headers, timeout=API_FALLBACK_TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return self._fallback_answer(query, context)
                
        except Exception as e:
            logger.error(f"Error with API fallback: {e}")
            return self._fallback_answer(query, context)
    
    def _fallback_answer(self, query: str, context: str) -> str:
        """Fallback answer when other methods fail - precise and concise"""
        query_lower = query.lower()
        
        # Extract relevance scores from context if available
        context_lines = context.split('\n')
        relevant_content = []

        for line in context_lines:
            if line.strip() and len(line.strip()) > 20:
                # Check if line has relevance score
                if '[Relevance:' in line:
                    # Extract content after relevance score
                    content = line.split('] ', 1)[-1] if '] ' in line else line
                    relevant_content.append(content.strip())
                else:
                    relevant_content.append(line.strip())

        # Find most relevant content based on query keywords
        query_words = [word.lower() for word in query.split() if len(word) > 3]
        scored_content = []

        for content in relevant_content:
            content_lower = content.lower()
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored_content.append((score, content))

        # Sort by relevance score
        scored_content.sort(key=lambda x: x[0], reverse=True)

        # Detect if question is in Arabic
        is_arabic = self._is_arabic_text(query)
        answer_parts = []
        
        if scored_content:
            # Take top 2-3 most relevant pieces
            top_content = [content for score, content in scored_content[:3]]

            # Check if user wants details
            wants_details = any(
                word in query_lower
                for word in ['explain', 'detail', 'how', 'why', 'what is', 'describe']
            )

            answer_parts.append(f"**{query}**\n")
            if wants_details:
                for i, content in enumerate(top_content, 1):
                    answer_parts.append(f"**{i}.** {content}")
            else:
                # Concise answer (2-3 lines)
                if len(top_content) >= 2:
                    answer_parts.append(f"• {top_content[0]}")
                    answer_parts.append(f"• {top_content[1]}")
                else:
                    answer_parts.append(f"• {top_content[0]}")

            # No language notes needed - respond in same language as question
                return "\n".join(answer_parts)
    
        else:
            return (
                f"❌ **Not Found in Documents**\n\n"
                f"I couldn't find relevant information in the uploaded PDFs or websites to answer: '{query}'. "
                "Please try rephrasing your question or add more relevant documents."
            )

    
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