"""
AI Response Generator Module
Handles all AI model interactions for generating responses
"""

import requests
import logging
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, OPENROUTER_TIMEOUT,
    API_FALLBACK_ENABLED, API_FALLBACK_URL, API_FALLBACK_MODEL, API_FALLBACK_TIMEOUT, API_KEY
)

logger = logging.getLogger(__name__)

class AIResponseGenerator:
    """Handles all AI model interactions for generating responses"""
    
    def __init__(self, language_processor):
        self.language_processor = language_processor
    
    def generate_rag_response(self, query: str, context: str) -> str:
        """Generate RAG response using available AI models"""
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
    
    def enhance_with_openrouter(self, query: str, rag_response: str, context: str) -> str:
        """Enhance RAG response using OpenRouter API"""
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
            is_arabic = self.language_processor.is_arabic_text(query)
            
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
    
    def _generate_ollama_answer(self, query: str, context: str) -> str:
        """Generate answer using Ollama"""
        try:
            # Check if user wants details
            wants_details = any(word in query.lower() for word in ['explain', 'detail', 'how', 'why', 'what is', 'describe'])
            
            # Detect if question is in Arabic
            is_arabic = self.language_processor.is_arabic_text(query)
            
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
    
    def _generate_api_answer(self, query: str, context: str) -> str:
        """Generate answer using API fallback"""
        if not API_FALLBACK_ENABLED or not API_KEY:
            return self._fallback_answer(query, context)
    
        try:
            # Check if user wants details
            wants_details = any(word in query.lower() for word in ['explain', 'detail', 'how', 'why', 'what is', 'describe'])
            
            # Detect if question is in Arabic
            is_arabic = self.language_processor.is_arabic_text(query)
            
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
        """Fallback answer when other methods fail"""
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
        is_arabic = self.language_processor.is_arabic_text(query)
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
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
