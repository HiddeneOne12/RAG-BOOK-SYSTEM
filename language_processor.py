"""
Language Processing Module
Handles language detection, translation, and conversion for the RAG system
"""

import requests
import logging
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

class LanguageProcessor:
    """Handles language detection and translation operations"""
    
    def __init__(self):
        self.arabic_unicode_ranges = [
            ('\u0600', '\u06FF'),  # Arabic
            ('\u0750', '\u077F'),  # Arabic Supplement
            ('\u08A0', '\u08FF'),  # Arabic Extended-A
            ('\uFB50', '\uFDFF'),  # Arabic Presentation Forms-A
            ('\uFE70', '\uFEFF'),  # Arabic Presentation Forms-B
        ]
    
    def is_arabic_text(self, text: str) -> bool:
        """Detect if text contains Arabic characters"""
        arabic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if self._is_arabic_char(char):
                    arabic_chars += 1
        
        # If more than 30% of alphabetic characters are Arabic, consider it Arabic text
        return total_chars > 0 and (arabic_chars / total_chars) > 0.3
    
    def _is_arabic_char(self, char: str) -> bool:
        """Check if character is in Arabic Unicode range"""
        for start, end in self.arabic_unicode_ranges:
            if start <= char <= end:
                return True
        return False
    
    def detect_document_language(self, collection) -> str:
        """Detect the primary language of documents in the collection"""
        try:
            # Get a sample of documents from the collection
            results = collection.get(limit=10)
            if not results['documents'] or len(results['documents']) == 0:
                return "english"  # Default to English
            
            # Analyze sample documents
            arabic_count = 0
            total_docs = len(results['documents'])
            
            for doc in results['documents']:
                if self.is_arabic_text(doc):
                    arabic_count += 1
            
            # If more than 50% of documents are Arabic, consider collection Arabic
            if arabic_count / total_docs > 0.5:
                return "arabic"
            else:
                return "english"
                
        except Exception as e:
            logger.error(f"Error detecting document language: {str(e)}")
            return "english"  # Default to English
    
    def convert_query_language(self, query: str, collection) -> str:
        """Convert query language to match document language"""
        try:
            # Detect document language
            doc_language = self.detect_document_language(collection)
            query_language = "arabic" if self.is_arabic_text(query) else "english"
            
            # If query is Arabic but documents are English, translate to English
            if query_language == "arabic" and doc_language == "english":
                return self.translate_arabic_to_english(query)
            # If query is English but documents are Arabic, translate to Arabic
            elif query_language == "english" and doc_language == "arabic":
                return self.translate_english_to_arabic(query)
            else:
                # Same language, return as is
                return query
                
        except Exception as e:
            logger.error(f"Error converting query language: {str(e)}")
            return query
    
    def translate_arabic_to_english(self, arabic_text: str) -> str:
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
    
    def translate_english_to_arabic(self, english_text: str) -> str:
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
