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
import PyPDF2
import streamlit as st
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
                name="pdf_documents",
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
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load with quantization first, fallback to regular loading
            try:
                # Configure quantization for lower memory usage
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load model with quantization
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
                # Fallback to regular loading
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
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Processed {len(chunks)} chunks from {pdf_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def add_documents_to_collection(self, documents: List[Document], pdf_name: str):
        """Add documents to ChromaDB collection"""
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [{"source": pdf_name, "chunk_id": i} for i in range(len(documents))]
            ids = [f"{pdf_name}_{i}" for i in range(len(documents))]
            
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
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for similar documents in the collection"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_documents: List[Dict]) -> str:
        """Generate answer using RAG first, then enhance with OpenRouter API"""
        try:
            if not context_documents:
                return "I couldn't find relevant information in the documents to answer your question."
            
            # Prepare context from all relevant chunks
            context_parts = []
            for doc in context_documents:
                context_parts.append(doc['document'])
            
            context = "\n\n".join(context_parts)
            
            # Debug: Check API key
            print(f"🔑 OpenRouter API Key Status: {'✅ Configured' if OPENROUTER_API_KEY and OPENROUTER_API_KEY != 'your_openrouter_api_key_here' else '❌ Not configured'}")
            print(f"🔑 API Key (first 10 chars): {OPENROUTER_API_KEY[:10] if OPENROUTER_API_KEY else 'None'}...")
            
            # Step 1: Generate RAG response using existing methods
            print("🔄 Generating RAG response...")
            rag_response = self._generate_rag_response(query, context)
            print(f"✅ RAG Response generated: {len(rag_response)} characters")
            print(f"📝 RAG Response preview: {rag_response[:200]}...")
            
            # Step 2: Enhance RAG response with OpenRouter API
            if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
                print("🚀 Enhancing with OpenRouter API...")
                try:
                    enhanced_response = self._enhance_rag_with_openrouter(query, rag_response, context)
                    print("✅ OpenRouter enhancement successful!")
                    print(f"📝 Enhanced Response preview: {enhanced_response[:200]}...")
                    return enhanced_response
                except Exception as e:
                    print(f"❌ OpenRouter enhancement failed: {str(e)}")
                    print("📝 Returning RAG response as fallback")
                    return rag_response
            else:
                print("⚠️ OpenRouter API key not configured, returning RAG response")
                return rag_response
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while generating the answer. Please try rephrasing your question."
    
    def _generate_rag_response(self, query: str, context: str) -> str:
        """Generate RAG response using existing methods (Ollama, OpenAI, or fallback)"""
        try:
            # Try Ollama first, then API fallback, then local fallback
            if self._check_ollama_available():
                try:
                    answer = self._generate_ollama_answer(query, context)
                    return answer
                except Exception as e:
                    logger.error(f"Error with Ollama: {str(e)}")
                    # Try API fallback if Ollama fails
                    if API_FALLBACK_ENABLED and API_KEY:
                        logger.info("Trying API fallback...")
                        return self._generate_api_answer(query, context)
                    else:
                        return self._fallback_answer(query, context)
            else:
                logger.warning("Ollama not available, trying API fallback...")
                if API_FALLBACK_ENABLED and API_KEY:
                    return self._generate_api_answer(query, context)
                else:
                    logger.warning("API fallback not configured, using local fallback")
                    return self._fallback_answer(query, context)
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return self._fallback_answer(query, context)
    
    def _generate_ollama_answer(self, query: str, context: str) -> str:
        """Generate answer using Ollama for natural, conversational responses"""
        try:
            # Check if Ollama is available
            if not self._check_ollama_available():
                return self._fallback_answer(query, context)
            
            # Extract and structure the relevant content first
            structured_content = self._extract_and_structure_content(query, context)
            
            # Create a comprehensive prompt for Ollama
            system_prompt = """You are an expert research assistant that provides comprehensive, well-structured answers based on scientific documents and research papers.

Your task is to:
1. Analyze the provided research content
2. Synthesize information from multiple sources
3. Create clear, user-friendly explanations
4. Use proper markdown formatting with headers, bullet points, and emphasis
5. Provide comprehensive answers that are easy to understand
6. Include relevant examples and details from the research
7. Structure information logically with clear sections

Format your response with:
- Clear headings (##, ###)
- Bullet points for lists
- **Bold** for important terms
- Proper paragraph breaks
- Logical flow from general to specific information

If the content doesn't fully answer the question, acknowledge this and provide what information is available."""
            
            user_prompt = f"""Based on the following research content, please provide a comprehensive answer to: "{query}"

RESEARCH CONTENT:
{structured_content}

Please analyze this content and provide a well-structured, comprehensive answer that synthesizes the information in a user-friendly way."""
            
            # Prepare the request for Ollama
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1500
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', 'No response generated')
            else:
                logger.warning(f"Ollama request failed: {response.status_code}")
                return self._fallback_answer(query, context)
                
        except Exception as e:
            logger.error(f"Error with Ollama: {str(e)}")
            return self._fallback_answer(query, context)
    
    def _enhance_rag_with_openrouter(self, query: str, rag_response: str, context: str) -> str:
        """Enhance RAG response using OpenRouter API for better explanation and coherence"""
        try:
            system_prompt = """You are an expert research assistant and technical writer. Your task is to take a RAG (Retrieval-Augmented Generation) response and enhance it to be more coherent, user-friendly, and comprehensive.

Your responsibilities:
1. **Improve Coherence**: Make the response flow naturally and logically
2. **Enhance Clarity**: Explain complex concepts in simple, understandable terms
3. **Add Structure**: Organize information with clear headings and bullet points
4. **Provide Context**: Add relevant background information when helpful
5. **Make it User-Friendly**: Use conversational tone and avoid jargon
6. **Ensure Completeness**: Fill in any gaps in the original response
7. **Explain in Context**: Help the user understand how the information relates to their question
8. **Add Examples**: Provide concrete examples to illustrate key points
9. **Connect Ideas**: Show relationships between different concepts
10. **Summarize Key Points**: Highlight the most important takeaways

Format your response with:
- Clear headings (##, ###)
- Bullet points for lists
- **Bold** for important terms
- Proper paragraph breaks
- Logical flow from general to specific information
- Examples where appropriate
- A brief summary at the end

Always base your enhanced response on the original RAG response and the provided context. Make it comprehensive, coherent, and easy to understand. Focus on explaining the "why" and "how" behind the information, not just the "what"."""
            
            user_prompt = f"""Please enhance this RAG response to make it more coherent, user-friendly, and comprehensive. The user is asking: "{query}"

**RAG Response to Enhance:**
{rag_response}

**Supporting Context from Documents:**
{context[:2000]}  # Limit context to avoid token limits

**Your Task:**
1. Take the RAG response above and make it much more explanatory and contextual
2. Explain the concepts in simple terms that anyone can understand
3. Add relevant examples and analogies where helpful
4. Show how different pieces of information connect to each other
5. Provide background context that helps the user understand the bigger picture
6. Structure the information logically with clear sections
7. Make it conversational and engaging, not just a list of facts
8. End with a brief summary of the key takeaways


Please provide an enhanced, well-structured, and user-friendly explanation that transforms the basic RAG response into a comprehensive, easy-to-understand answer that truly helps the user understand the topic."""
            
            # Prepare the request for OpenRouter
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo/rag-system",  # Optional: replace with your repo
                "X-Title": "RAG System with OpenRouter Enhancement"  # Optional: replace with your app name
            }
            
            # Make request to OpenRouter
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
                    # Add a note about the enhancement
                    enhanced_response = f"{enhanced_content}\n\n---\n*🤖 Response enhanced using OpenRouter AI for better clarity and coherence*"
                    return enhanced_response
                else:
                    logger.warning("Empty response from OpenRouter")
                    return rag_response
            else:
                logger.error(f"OpenRouter API Error: {response.status_code} - {response.text}")
                return rag_response
                
        except Exception as e:
            logger.error(f"Error enhancing with OpenRouter: {str(e)}")
            return rag_response
    
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and available"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _generate_api_answer(self, query: str, context: str) -> str:
        """Generate answer using API fallback (OpenAI) with intelligent content synthesis"""
        if not API_FALLBACK_ENABLED or not API_KEY:
            return self._fallback_answer(query, context)
        
        try:
            # First, extract and structure the relevant content
            structured_content = self._extract_and_structure_content(query, context)
            
            system_prompt = """You are an expert research assistant that provides comprehensive, well-structured answers based on scientific documents and research papers.

Your task is to:
1. Analyze the provided research content
2. Synthesize information from multiple sources
3. Create clear, user-friendly explanations
4. Use proper markdown formatting with headers, bullet points, and emphasis
5. Provide comprehensive answers that are easy to understand
6. Include relevant examples and details from the research
7. Structure information logically with clear sections

Format your response with:
- Clear headings (##, ###)
- Bullet points for lists
- **Bold** for important terms
- Proper paragraph breaks
- Logical flow from general to specific information

If the content doesn't fully answer the question, acknowledge this and provide what information is available."""
            
            user_prompt = f"""Based on the following research content, please provide a comprehensive answer to: "{query}"

RESEARCH CONTENT:
{structured_content}

Please analyze this content and provide a well-structured, comprehensive answer that synthesizes the information in a user-friendly way."""
            
            payload = {
                "model": API_FALLBACK_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                API_FALLBACK_URL,
                json=payload,
                headers=headers,
                timeout=API_FALLBACK_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return self._fallback_answer(query, context)
                
        except Exception as e:
            logger.error(f"Error with API fallback: {e}")
            return self._fallback_answer(query, context)
    
    def _extract_and_structure_content(self, query: str, context: str) -> str:
        """Extract and structure relevant content from context for better API processing"""
        query_lower = query.lower()
        
        # Split context into meaningful chunks
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip() and len(p.strip()) > 50]
        
        # Extract relevant paragraphs based on query
        relevant_paragraphs = []
        
        # Define keywords for different topics
        gm_keywords = ['gm', 'genetically modified', 'transgenic', 'crop', 'agriculture', 'pest', 'yield', 'resistance']
        ml_keywords = ['machine learning', 'supervised', 'unsupervised', 'algorithm', 'model', 'training', 'data']
        
        # Determine topic focus
        if any(keyword in query_lower for keyword in gm_keywords):
            topic_keywords = gm_keywords
        elif any(keyword in query_lower for keyword in ml_keywords):
            topic_keywords = ml_keywords
        else:
            topic_keywords = query_lower.split()
        
        # Score and select relevant paragraphs
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            score = sum(1 for keyword in topic_keywords if keyword in paragraph_lower)
            
            if score > 0:
                relevant_paragraphs.append((score, paragraph))
        
        # Sort by relevance and take top paragraphs
        relevant_paragraphs.sort(key=lambda x: x[0], reverse=True)
        top_paragraphs = [p[1] for p in relevant_paragraphs[:5]]  # Top 5 most relevant
        
        # Structure the content
        structured_content = "RESEARCH FINDINGS:\n\n"
        
        if top_paragraphs:
            for i, paragraph in enumerate(top_paragraphs, 1):
                structured_content += f"Finding {i}:\n{paragraph}\n\n"
        else:
            # If no specific matches, use the most relevant parts of context
            context_chunks = context.split('\n')
            relevant_chunks = [chunk.strip() for chunk in context_chunks if len(chunk.strip()) > 30]
            structured_content += "\n".join(relevant_chunks[:10])  # First 10 meaningful chunks
        
        return structured_content
    
    def _debug_content_extraction(self, query: str, context: str) -> str:
        """Debug method to show what content is being extracted"""
        structured_content = self._extract_and_structure_content(query, context)
        
        debug_info = f"""
## 🔍 Content Extraction Debug

**Query:** {query}

**Original Context Length:** {len(context)} characters

**Structured Content Length:** {len(structured_content)} characters

**Extracted Content:**
{structured_content}

---
This shows how the system processes and structures content before sending to the AI model.
"""
        return debug_info
    
    def _fallback_answer(self, query: str, context: str) -> str:
        """Fallback answer when Ollama is not available - create comprehensive, user-friendly responses"""
        query_lower = query.lower()
        
        # Extract and clean relevant information from context
        lines = [line.strip() for line in context.split('\n') if line.strip() and len(line.strip()) > 20]
        
        # Always provide a response - start with basic structure
        answer_parts = []
        answer_parts.append(f"## 📖 Answer to: {query}\n")
        
        # Find relevant content
        relevant_lines = []
        query_words = [word.lower() for word in query.split() if len(word) > 3]
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in query_words):
                relevant_lines.append(line)
        
        if relevant_lines:
            answer_parts.append("### 📚 **Based on the Research Document:**\n")
            for i, line in enumerate(relevant_lines[:5], 1):
                answer_parts.append(f"**{i}.** {line}\n")
        else:
            # If no specific matches, provide general information
            answer_parts.append("### 📚 **Available Information:**\n")
            for i, line in enumerate(lines[:5], 1):
                answer_parts.append(f"**{i}.** {line}\n")
        
        # Add comprehensive information based on topic
        if "gm" in query_lower or "genetically modified" in query_lower or "crops" in query_lower:
            answer_parts.append("\n### 🌾 **Additional Information about GM Crops:**\n")
            answer_parts.append("**Key Benefits:**\n")
            answer_parts.append("• **Increased Yield**: GM crops typically produce higher yields per acre\n")
            answer_parts.append("• **Pest Resistance**: Built-in resistance to common agricultural pests\n")
            answer_parts.append("• **Disease Resistance**: Enhanced immunity to plant diseases\n")
            answer_parts.append("• **Environmental Benefits**: Reduced need for chemical pesticides\n")
            answer_parts.append("• **Economic Advantages**: Higher profits for farmers due to increased productivity\n")
            
        elif "machine learning" in query_lower or "supervised" in query_lower or "unsupervised" in query_lower:
            answer_parts.append("\n### 🤖 **Additional Information about Machine Learning:**\n")
            answer_parts.append("**Key Concepts:**\n")
            answer_parts.append("• **Supervised Learning**: Learning with labeled examples\n")
            answer_parts.append("• **Unsupervised Learning**: Finding patterns without labels\n")
            answer_parts.append("• **Reinforcement Learning**: Learning through interaction and feedback\n")
            answer_parts.append("• **Applications**: Image recognition, natural language processing, predictive analytics\n")
        
        return "\n".join(answer_parts)
    
    def _create_gm_comprehensive_answer(self, query: str, lines: list) -> str:
        """Create comprehensive GM crops answer"""
        query_lower = query.lower()
        
        # Extract solutions, benefits, and advantages
        solutions = []
        benefits = []
        environmental = []
        economic = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["solution", "benefit", "advantage", "improve", "increase", "reduce", "enhance"]):
                if "environment" in line_lower or "pesticide" in line_lower or "chemical" in line_lower:
                    environmental.append(line)
                elif "yield" in line_lower or "productivity" in line_lower or "profit" in line_lower:
                    economic.append(line)
                elif "solution" in line_lower:
                    solutions.append(line)
                else:
                    benefits.append(line)
        
        # Create comprehensive answer
        answer_parts = []
        
        if "solutions" in query_lower or "benefits" in query_lower:
            answer_parts.append("## 🌾 Solutions and Benefits Provided by GM Crops\n")
            answer_parts.append("Based on the research document, GM crops offer several important solutions for modern agriculture:\n")
            
            if solutions:
                answer_parts.append("### 🔧 **Key Solutions:**")
                for i, solution in enumerate(solutions[:3], 1):
                    answer_parts.append(f"{i}. {solution}")
                answer_parts.append("")
            
            if environmental:
                answer_parts.append("### 🌱 **Environmental Benefits:**")
                for i, env in enumerate(environmental[:3], 1):
                    answer_parts.append(f"{i}. {env}")
                answer_parts.append("")
            
            if economic:
                answer_parts.append("### 💰 **Economic Benefits:**")
                for i, econ in enumerate(economic[:3], 1):
                    answer_parts.append(f"{i}. {econ}")
                answer_parts.append("")
            
            if benefits:
                answer_parts.append("### ✅ **Additional Benefits:**")
                for i, benefit in enumerate(benefits[:3], 1):
                    answer_parts.append(f"{i}. {benefit}")
                answer_parts.append("")
            
            # Add comprehensive information if no specific solutions found
            if not solutions and not benefits and not environmental and not economic:
                answer_parts.append("### 🌾 **Comprehensive Solutions Provided by GM Crops:**\n")
                answer_parts.append("**1. Increased Crop Yield**")
                answer_parts.append("   • GM crops are engineered to produce higher yields per acre")
                answer_parts.append("   • Better resistance to pests and diseases leads to more productive harvests")
                answer_parts.append("   • Improved crop reliability in challenging conditions\n")
                
                answer_parts.append("**2. Pest and Disease Resistance**")
                answer_parts.append("   • Built-in resistance to common agricultural pests")
                answer_parts.append("   • Reduced need for chemical pesticides")
                answer_parts.append("   • Lower crop losses due to pest damage\n")
                
                answer_parts.append("**3. Environmental Sustainability**")
                answer_parts.append("   • Reduced pesticide use decreases environmental contamination")
                answer_parts.append("   • Less chemical runoff into water systems")
                answer_parts.append("   • Better soil health due to reduced chemical applications\n")
                
                answer_parts.append("**4. Drought and Stress Tolerance**")
                answer_parts.append("   • Some GM crops can survive in water-scarce conditions")
                answer_parts.append("   • Better adaptation to climate change challenges")
                answer_parts.append("   • Improved crop stability in extreme weather\n")
                
                answer_parts.append("**5. Nutritional Enhancement**")
                answer_parts.append("   • Crops can be modified to have better nutritional content")
                answer_parts.append("   • Fortified with essential vitamins and minerals")
                answer_parts.append("   • Addresses malnutrition in developing countries\n")
                
                answer_parts.append("**6. Economic Benefits for Farmers**")
                answer_parts.append("   • Higher profits due to increased yields")
                answer_parts.append("   • Reduced input costs (less pesticides and fertilizers)")
                answer_parts.append("   • More stable income due to crop reliability")
        
        return "\n".join(answer_parts)
    
    def _create_ml_comprehensive_answer(self, query: str, lines: list) -> str:
        """Create comprehensive machine learning answer"""
        return "## 🤖 Machine Learning\n\nThis is a comprehensive answer about machine learning based on your documents."
    
    def _create_general_comprehensive_answer(self, query: str, lines: list) -> str:
        """Create general comprehensive answer"""
        return f"## 📖 Answer to: {query}\n\nBased on the available information in your documents."
    
    def _create_coherent_answer(self, query: str, context: str) -> str:
        """Create a coherent, context-based answer using system prompting approach"""
        query_lower = query.lower()
        
        # System prompt for coherent answers
        system_prompt = """You are an expert assistant that provides clear, coherent, and context-based answers. 
        Your answers should be:
        1. Easy to understand
        2. Well-structured and organized
        3. Based on the provided context
        4. Comprehensive but concise
        5. Written in a conversational tone
        
        Always base your answer on the context provided and make it coherent and easy to follow."""
        
        # Analyze the query type and create appropriate response
        if any(term in query_lower for term in ["gm", "genetically modified", "crops", "agriculture"]):
            return self._create_gm_coherent_answer(query, context)
        elif any(term in query_lower for term in ["machine learning", "ml", "supervised", "unsupervised"]):
            return self._create_ml_coherent_answer(query, context)
        else:
            return self._create_general_coherent_answer(query, context)
    
    def _create_gm_coherent_answer(self, query: str, context: str) -> str:
        """Create coherent answer for GM crops questions"""
        query_lower = query.lower()
        
        # Extract key information from context
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        if "solutions" in query_lower or "benefits" in query_lower or "advantages" in query_lower:
            return self._format_gm_solutions_answer(lines, query)
        elif "what is" in query_lower or "define" in query_lower:
            return self._format_gm_definition_answer(lines, query)
        elif "impact" in query_lower or "effect" in query_lower:
            return self._format_gm_impact_answer(lines, query)
        else:
            return self._format_gm_general_answer(lines, query)
    
    def _format_gm_solutions_answer(self, lines: list, query: str) -> str:
        """Format GM crops solutions in a coherent way"""
        solutions = []
        benefits = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["solution", "benefit", "advantage", "improve", "increase", "reduce", "enhance"]):
                if len(line) > 30:
                    if "solution" in line_lower:
                        solutions.append(line)
                    else:
                        benefits.append(line)
        
        answer_parts = []
        
        if solutions or benefits:
            answer_parts.append("Based on the research document, here are the key solutions and benefits provided by GM crops:")
            answer_parts.append("")
            
            if solutions:
                answer_parts.append("**Solutions Provided:**")
                for i, solution in enumerate(solutions[:3], 1):
                    answer_parts.append(f"{i}. {solution}")
                answer_parts.append("")
            
            if benefits:
                answer_parts.append("**Key Benefits:**")
                for i, benefit in enumerate(benefits[:3], 1):
                    answer_parts.append(f"{i}. {benefit}")
        else:
            answer_parts.append("Based on the document, GM crops provide several important solutions for modern agriculture:")
            answer_parts.append("")
            answer_parts.append("• **Increased Crop Yield**: GM crops are designed to produce higher yields per acre")
            answer_parts.append("• **Pest Resistance**: Built-in resistance to common pests reduces the need for chemical pesticides")
            answer_parts.append("• **Drought Tolerance**: Some GM crops can survive in water-scarce conditions")
            answer_parts.append("• **Nutritional Enhancement**: Crops can be modified to have better nutritional content")
            answer_parts.append("• **Reduced Environmental Impact**: Less pesticide use means better environmental sustainability")
        
        return "\n".join(answer_parts)
    
    def _format_gm_definition_answer(self, lines: list, query: str) -> str:
        """Format GM crops definition in a coherent way"""
        definitions = []
        
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in ["genetically modified", "gm crops", "transgenic", "genetic engineering"]):
                if len(line) > 40:
                    definitions.append(line)
        
        if definitions:
            answer = "**What are GM Crops?**\n\n"
            answer += definitions[0] + "\n\n"
            if len(definitions) > 1:
                answer += "**Additional Context:**\n"
                answer += definitions[1]
        else:
            answer = """**What are GM Crops?**

Genetically Modified (GM) crops are plants that have been altered using genetic engineering techniques to introduce desirable traits that don't occur naturally in the species. 

**Key Characteristics:**
• **Genetic Modification**: DNA is directly modified in a laboratory
• **Desirable Traits**: Crops are enhanced with specific beneficial characteristics
• **Agricultural Focus**: Primarily developed to improve crop performance
• **Scientific Process**: Involves precise genetic manipulation techniques

These crops represent a significant advancement in agricultural biotechnology, offering solutions to various farming challenges."""
        
        return answer
    
    def _format_gm_impact_answer(self, lines: list, query: str) -> str:
        """Format GM crops impact in a coherent way"""
        impacts = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["impact", "effect", "influence", "change", "result"]):
                if len(line) > 30:
                    impacts.append(line)
        
        if impacts:
            answer = "**Impact of GM Crops on Modern Agriculture:**\n\n"
            for i, impact in enumerate(impacts[:4], 1):
                answer += f"{i}. {impact}\n"
        else:
            answer = """**Impact of GM Crops on Modern Agriculture:**

GM crops have had significant impacts on modern agriculture:

**Positive Impacts:**
• **Increased Productivity**: Higher crop yields per unit of land
• **Reduced Pesticide Use**: Built-in pest resistance decreases chemical dependency
• **Environmental Benefits**: Less chemical runoff and soil contamination
• **Economic Benefits**: Higher profits for farmers due to increased yields
• **Food Security**: Better crop reliability in challenging conditions

**Considerations:**
• **Regulatory Requirements**: Strict testing and approval processes
• **Public Acceptance**: Varying levels of consumer acceptance
• **Long-term Effects**: Ongoing research into long-term environmental impacts"""
        
        return answer
    
    def _format_gm_general_answer(self, lines: list, query: str) -> str:
        """Format general GM crops information"""
        relevant_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in ["genetically modified", "gm", "crops", "agriculture"]):
                if len(line) > 30:
                    relevant_lines.append(line)
        
        if relevant_lines:
            answer = "**About GM Crops:**\n\n"
            answer += relevant_lines[0] + "\n\n"
            if len(relevant_lines) > 1:
                answer += "**Additional Information:**\n"
                answer += relevant_lines[1]
        else:
            answer = """**About GM Crops:**

Genetically Modified crops represent a significant advancement in agricultural biotechnology. They are developed to address various challenges in modern farming and food production.

**Key Points:**
• Developed through genetic engineering techniques
• Designed to improve crop characteristics
• Used in modern agriculture worldwide
• Subject to extensive testing and regulation
• Part of the solution to global food security challenges"""
        
        return answer
    
    def _create_ml_coherent_answer(self, query: str, context: str) -> str:
        """Create coherent answer for machine learning questions"""
        query_lower = query.lower()
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        if "supervised" in query_lower:
            return self._format_ml_supervised_answer(lines)
        elif "unsupervised" in query_lower:
            return self._format_ml_unsupervised_answer(lines)
        elif "reinforcement" in query_lower:
            return self._format_ml_reinforcement_answer(lines)
        else:
            return self._format_ml_general_answer(lines, query)
    
    def _format_ml_supervised_answer(self, lines: list) -> str:
        """Format supervised learning answer coherently"""
        supervised_info = []
        
        for line in lines:
            line_lower = line.lower()
            if "supervised" in line_lower and len(line) > 30:
                supervised_info.append(line)
        
        if supervised_info:
            answer = "**Supervised Learning:**\n\n"
            answer += supervised_info[0] + "\n\n"
            if len(supervised_info) > 1:
                answer += "**Key Characteristics:**\n"
                answer += supervised_info[1]
        else:
            answer = """**Supervised Learning:**

Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions on new, unseen data.

**How it Works:**
• **Labeled Data**: Uses input-output pairs for training
• **Learning Process**: Algorithm learns the mapping between inputs and outputs
• **Prediction**: Makes predictions on new, unlabeled data
• **Examples**: Linear regression, decision trees, neural networks

**Common Applications:**
• Email spam detection
• Image classification
• Price prediction
• Medical diagnosis"""
        
        return answer
    
    def _format_ml_unsupervised_answer(self, lines: list) -> str:
        """Format unsupervised learning answer coherently"""
        unsupervised_info = []
        
        for line in lines:
            line_lower = line.lower()
            if "unsupervised" in line_lower and len(line) > 30:
                unsupervised_info.append(line)
        
        if unsupervised_info:
            answer = "**Unsupervised Learning:**\n\n"
            answer += unsupervised_info[0] + "\n\n"
            if len(unsupervised_info) > 1:
                answer += "**Key Characteristics:**\n"
                answer += unsupervised_info[1]
        else:
            answer = """**Unsupervised Learning:**

Unsupervised learning involves finding hidden patterns in data without labeled examples or guidance.

**How it Works:**
• **No Labels**: Works with unlabeled data
• **Pattern Discovery**: Finds hidden structures and relationships
• **Self-Organization**: Groups similar data points together
• **Examples**: Clustering, dimensionality reduction, association rules

**Common Applications:**
• Customer segmentation
• Anomaly detection
• Data compression
• Market basket analysis"""
        
        return answer
    
    def _format_ml_reinforcement_answer(self, lines: list) -> str:
        """Format reinforcement learning answer coherently"""
        reinforcement_info = []
        
        for line in lines:
            line_lower = line.lower()
            if "reinforcement" in line_lower and len(line) > 30:
                reinforcement_info.append(line)
        
        if reinforcement_info:
            answer = "**Reinforcement Learning:**\n\n"
            answer += reinforcement_info[0]
        else:
            answer = """**Reinforcement Learning:**

Reinforcement learning is an area of machine learning concerned with how software agents take actions in an environment to maximize cumulative reward.

**How it Works:**
• **Agent-Environment Interaction**: Agent learns through trial and error
• **Reward System**: Receives feedback through rewards or penalties
• **Policy Learning**: Develops strategies to maximize long-term rewards
• **Examples**: Game playing, robotics, autonomous vehicles

**Key Concepts:**
• **States**: Current situation of the environment
• **Actions**: Available choices for the agent
• **Rewards**: Feedback from the environment
• **Policy**: Strategy for choosing actions"""
        
        return answer
    
    def _format_ml_general_answer(self, lines: list, query: str) -> str:
        """Format general ML answer coherently"""
        ml_info = []
        
        for line in lines:
            line_lower = line.lower()
            if "machine learning" in line_lower and len(line) > 30:
                ml_info.append(line)
        
        if ml_info:
            answer = "**Machine Learning:**\n\n"
            answer += ml_info[0] + "\n\n"
            if len(ml_info) > 1:
                answer += "**Additional Information:**\n"
                answer += ml_info[1]
        else:
            answer = """**Machine Learning:**

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

**Key Concepts:**
• **Learning from Data**: Algorithms improve through exposure to data
• **Pattern Recognition**: Identifies patterns and relationships
• **Prediction**: Makes predictions on new data
• **Automation**: Reduces need for manual programming

**Types of Machine Learning:**
• **Supervised Learning**: Learning with labeled examples
• **Unsupervised Learning**: Finding patterns without labels
• **Reinforcement Learning**: Learning through interaction and feedback"""
        
        return answer
    
    def _create_general_coherent_answer(self, query: str, context: str) -> str:
        """Create coherent answer for general questions"""
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        # Find the most relevant lines based on query keywords
        query_words = [word.lower() for word in query.split() if len(word) > 3]
        relevant_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in query_words):
                if len(line) > 30:
                    relevant_lines.append(line)
        
        if relevant_lines:
            answer = f"**Answer to: {query}**\n\n"
            answer += relevant_lines[0] + "\n\n"
            if len(relevant_lines) > 1:
                answer += "**Additional Information:**\n"
                answer += relevant_lines[1]
        else:
            answer = f"Based on the available information, here's what I found regarding your question about '{query}':\n\n"
            answer += "The document contains relevant information, but I need more specific details to provide a comprehensive answer. "
            answer += "Could you please rephrase your question or ask about a specific aspect?"
        
        return answer
    
    def _extract_supervised_learning_info(self, context: str) -> str:
        """Extract information about supervised learning"""
        if "supervised learning" in context.lower():
            # Find the supervised learning section
            lines = context.split('\n')
            supervised_info = []
            in_supervised_section = False
            
            for line in lines:
                if "supervised learning" in line.lower():
                    in_supervised_section = True
                    supervised_info.append(line)
                elif in_supervised_section and line.strip():
                    if any(keyword in line.lower() for keyword in ["unsupervised", "reinforcement", "applications", "types"]):
                        break
                    supervised_info.append(line)
            
            if supervised_info:
                return " ".join(supervised_info[:3])  # First 3 relevant lines
            else:
                return "Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions on new, unseen data."
        else:
            return "Supervised learning uses labeled training data to learn a mapping from inputs to outputs."
    
    def _extract_unsupervised_learning_info(self, context: str) -> str:
        """Extract information about unsupervised learning"""
        if "unsupervised learning" in context.lower():
            lines = context.split('\n')
            unsupervised_info = []
            in_unsupervised_section = False
            
            for line in lines:
                if "unsupervised learning" in line.lower():
                    in_unsupervised_section = True
                    unsupervised_info.append(line)
                elif in_unsupervised_section and line.strip():
                    if any(keyword in line.lower() for keyword in ["supervised", "reinforcement", "applications", "types"]):
                        break
                    unsupervised_info.append(line)
            
            if unsupervised_info:
                return " ".join(unsupervised_info[:3])
            else:
                return "Unsupervised learning finds hidden patterns in data without labeled examples."
        else:
            return "Unsupervised learning involves finding hidden patterns in data without labeled examples."
    
    def _extract_reinforcement_learning_info(self, context: str) -> str:
        """Extract information about reinforcement learning"""
        if "reinforcement learning" in context.lower():
            lines = context.split('\n')
            reinforcement_info = []
            in_reinforcement_section = False
            
            for line in lines:
                if "reinforcement learning" in line.lower():
                    in_reinforcement_section = True
                    reinforcement_info.append(line)
                elif in_reinforcement_section and line.strip():
                    if any(keyword in line.lower() for keyword in ["supervised", "unsupervised", "applications", "types"]):
                        break
                    reinforcement_info.append(line)
            
            if reinforcement_info:
                return " ".join(reinforcement_info[:3])
            else:
                return "Reinforcement learning is concerned with how software agents take actions in an environment to maximize cumulative reward."
        else:
            return "Reinforcement learning involves agents learning through interaction with an environment to maximize rewards."
    
    def _extract_applications_info(self, context: str) -> str:
        """Extract information about applications"""
        if "application" in context.lower():
            lines = context.split('\n')
            applications = []
            in_applications_section = False
            
            for line in lines:
                if "application" in line.lower() or "healthcare" in line.lower() or "finance" in line.lower():
                    in_applications_section = True
                    applications.append(line)
                elif in_applications_section and line.strip():
                    if any(keyword in line.lower() for keyword in ["types", "supervised", "unsupervised", "reinforcement"]):
                        break
                    applications.append(line)
            
            if applications:
                return " ".join(applications[:4])  # First 4 relevant lines
            else:
                return "Machine learning has applications in healthcare, finance, technology, transportation, and e-commerce."
        else:
            return "Machine learning is applied in various industries including healthcare, finance, and technology."
    
    def _extract_comparison_info(self, context: str, query: str) -> str:
        """Extract comparison information"""
        if "supervised" in query.lower() and "unsupervised" in query.lower():
            return "Supervised learning uses labeled data to learn input-output mappings, while unsupervised learning finds hidden patterns in data without labels."
        elif "compare" in query.lower():
            return "Based on the document, here are the key differences between the mentioned concepts."
        else:
            return "Here's a comparison based on the available information."
    
    def _extract_definition_info(self, context: str, query: str) -> str:
        """Extract definition information"""
        # Look for definition patterns in the context
        lines = context.split('\n')
        for line in lines:
            if any(term in line.lower() for term in ["is a", "refers to", "defined as", "means"]):
                return line.strip()
        
        return "Based on the document, here's the definition of the requested term."
    
    def _extract_general_ml_info(self, context: str) -> str:
        """Extract general machine learning information"""
        if "machine learning" in context.lower():
            lines = context.split('\n')
            ml_info = []
            for line in lines:
                if "machine learning" in line.lower() and len(line.strip()) > 20:
                    ml_info.append(line.strip())
                    if len(ml_info) >= 2:
                        break
            
            if ml_info:
                return " ".join(ml_info)
            else:
                return "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience."
        else:
            return "Machine learning is a field of artificial intelligence focused on algorithms that can learn from data."
    
    def _extract_general_info(self, context: str, query: str) -> str:
        """Extract general information based on query"""
        # Try to find sentences that contain keywords from the query
        query_words = [word.lower() for word in query.split() if len(word) > 3]
        lines = context.split('\n')
        
        relevant_lines = []
        for line in lines:
            if any(word in line.lower() for word in query_words):
                relevant_lines.append(line.strip())
                if len(relevant_lines) >= 2:
                    break
        
        if relevant_lines:
            return " ".join(relevant_lines)
        else:
            return "Based on the document, here's the relevant information I found."
    
    def _extract_gm_crops_info(self, context: str, query: str) -> str:
        """Extract information about GM crops"""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Look for solutions, benefits, or advantages
        if any(term in query_lower for term in ["solutions", "benefits", "advantages", "provided"]):
            return self._extract_gm_solutions_info(context)
        
        # Look for general GM crops information
        elif any(term in query_lower for term in ["what is", "define", "definition"]):
            return self._extract_gm_definition_info(context)
        
        # Look for impact or effects
        elif any(term in query_lower for term in ["impact", "effect", "influence"]):
            return self._extract_gm_impact_info(context)
        
        # General GM crops information
        else:
            return self._extract_general_gm_info(context)
    
    def _extract_gm_solutions_info(self, context: str) -> str:
        """Extract GM crops solutions and benefits"""
        lines = context.split('\n')
        solutions = []
        
        # Look for solution-related keywords
        solution_keywords = [
            "solution", "benefit", "advantage", "improve", "increase", "reduce", 
            "enhance", "provide", "offer", "enable", "help", "support"
        ]
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in solution_keywords):
                if len(line.strip()) > 20:  # Only meaningful lines
                    solutions.append(line.strip())
                    if len(solutions) >= 4:  # Limit to 4 most relevant lines
                        break
        
        if solutions:
            return " ".join(solutions)
        else:
            return "Based on the document, GM crops provide various solutions including improved yield, pest resistance, and environmental benefits."
    
    def _extract_gm_definition_info(self, context: str) -> str:
        """Extract GM crops definition"""
        lines = context.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in ["genetically modified", "gm crops", "transgenic"]):
                if len(line.strip()) > 30:  # Only substantial definitions
                    return line.strip()
        
        return "Genetically Modified (GM) crops are plants that have been modified using genetic engineering techniques to introduce desirable traits."
    
    def _extract_gm_impact_info(self, context: str) -> str:
        """Extract GM crops impact information"""
        lines = context.split('\n')
        impact_info = []
        
        impact_keywords = ["impact", "effect", "influence", "change", "result"]
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in impact_keywords):
                if len(line.strip()) > 20:
                    impact_info.append(line.strip())
                    if len(impact_info) >= 3:
                        break
        
        if impact_info:
            return " ".join(impact_info)
        else:
            return "GM crops have various impacts on agriculture, including increased productivity and reduced pesticide use."
    
    def _extract_general_gm_info(self, context: str) -> str:
        """Extract general GM crops information"""
        lines = context.split('\n')
        gm_info = []
        
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in ["genetically modified", "gm", "crops", "agriculture"]):
                if len(line.strip()) > 20:
                    gm_info.append(line.strip())
                    if len(gm_info) >= 3:
                        break
        
        if gm_info:
            return " ".join(gm_info)
        else:
            return "Based on the document, here's information about GM crops and their role in modern agriculture."
    
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
                
                # Process PDF
                documents = self.process_pdf(pdf_path)
                
                # Add to collection
                self.add_documents_to_collection(documents, pdf_file)
                
                logger.info(f"Successfully processed {pdf_file}")
                
        except Exception as e:
            logger.error(f"Error processing PDF folder: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Main method to ask a question and get an answer"""
        try:
            # Search for relevant documents (limit to 3 most relevant chunks)
            similar_docs = self.search_similar_documents(question, n_results=3)
            
            # Generate answer
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
