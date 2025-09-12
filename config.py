import os
from dotenv import load_dotenv

load_dotenv()

# Configuration settings
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # Lightweight model for your system

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "30"))

# API Fallback Configuration (OpenAI as backup)
API_FALLBACK_ENABLED = os.getenv("API_FALLBACK_ENABLED", "True").lower() == "true"
API_FALLBACK_URL = os.getenv("API_FALLBACK_URL", "https://api.openai.com/v1/chat/completions")
API_FALLBACK_MODEL = os.getenv("API_FALLBACK_MODEL", "gpt-3.5-turbo")
API_FALLBACK_TIMEOUT = int(os.getenv("API_FALLBACK_TIMEOUT", "5"))
API_KEY = os.getenv("OPENAI_API_KEY", "")  # Add your API key to .env file

# Create necessary directories
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
