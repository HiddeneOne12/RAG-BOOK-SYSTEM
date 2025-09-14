# RAG System Technical Documentation

## 🏗️ System Architecture

The RAG (Retrieval-Augmented Generation) system is built with a modular architecture for better maintainability and scalability.

### Core Modules

1. **`rag_system.py`** (Main Module - 200 lines)
   - Main RAG system class
   - Orchestrates all other modules
   - Handles question-answering workflow

2. **`language_processor.py`** (Language Processing - 150 lines)
   - Arabic/English text detection
   - Document language detection
   - Query translation using OpenRouter API

3. **`ai_response_generator.py`** (AI Response Generation - 300 lines)
   - Ollama integration
   - OpenRouter API enhancement
   - API fallback mechanisms
   - Fallback answer generation

4. **`document_processor.py`** (Document Processing - 100 lines)
   - PDF processing with PyPDFLoader
   - Website processing with BeautifulSoup
   - Text chunking with RecursiveCharacterTextSplitter

5. **`vector_database.py`** (Vector Database Management - 120 lines)
   - ChromaDB operations
   - Embedding generation with SentenceTransformer
   - Vector similarity search

6. **`content_manager.py`** (Content Management - 117 lines)
   - Orchestrates document addition
   - Website processing coordination
   - PDF folder processing

7. **`website_processor.py`** (Website Processing - 150 lines)
   - Web scraping with requests
   - HTML parsing with BeautifulSoup
   - URL validation and content extraction

## 📄 PDF Processing Details

### How PDF Processing Works

**Step 1: PDF Loading**
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(pdf_path)
documents = loader.load()
```
- **Package**: `langchain_community.document_loaders.PyPDFLoader`
- **Purpose**: Extracts text content from PDF files
- **Output**: List of Document objects with page content and metadata

**Step 2: Text Chunking**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```
- **Package**: `langchain.text_splitter.RecursiveCharacterTextSplitter`
- **Purpose**: Splits large documents into smaller, manageable chunks
- **Parameters**:
  - `chunk_size`: 1000 characters per chunk
  - `chunk_overlap`: 200 characters overlap between chunks
  - `separators`: Priority order for splitting (paragraphs → lines → words → characters)

**Step 3: Vector Embedding**
```python
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(texts).tolist()
```
- **Package**: `sentence_transformers.SentenceTransformer`
- **Model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Purpose**: Converts text chunks to numerical vectors for similarity search

**Step 4: Vector Storage**
```python
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
```
- **Package**: `chromadb`
- **Purpose**: Stores vector embeddings for fast similarity search
- **Distance Metric**: Cosine similarity

## 🌐 Website Processing Details

### How Website Processing Works

**Step 1: HTTP Request**
```python
import requests
response = requests.get(url, timeout=10)
response.raise_for_status()
```
- **Package**: `requests`
- **Purpose**: Fetches webpage content via HTTP GET request
- **Timeout**: 10 seconds to prevent hanging

**Step 2: HTML Parsing**
```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Remove script and style elements
for script in soup(["script", "style"]):
    script.decompose()

# Extract text content
text = soup.get_text()
```
- **Package**: `beautifulsoup4` with `lxml` backend
- **Purpose**: Parses HTML and extracts clean text content
- **Cleaning**: Removes JavaScript, CSS, and other non-content elements

**Step 3: Text Normalization**
```python
import re
clean_content = re.sub(r'\s+', ' ', text).strip()
```
- **Package**: `re` (built-in)
- **Purpose**: Normalizes whitespace and removes extra spaces

**Step 4: Document Creation and Chunking**
```python
from langchain.schema import Document
doc = Document(page_content=clean_content, metadata={"source": url, "type": "website"})
chunks = text_splitter.split_documents([doc])
```
- **Package**: `langchain.schema.Document`
- **Purpose**: Creates structured document with metadata
- **Chunking**: Same process as PDF processing

## 🔍 Vector Search Process

### How Vector Similarity Search Works

**Step 1: Query Embedding**
```python
query_embedding = embedding_model.encode([query]).tolist()[0]
```
- Converts user question to vector using same embedding model
- Ensures query and documents are in same vector space

**Step 2: Similarity Search**
```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=['documents', 'metadatas', 'distances']
)
```
- Searches ChromaDB for most similar document vectors
- Returns documents, metadata, and distance scores

**Step 3: Relevance Filtering**
```python
if results['distances'][0][i] < 0.8:
    relevance_score = 1 - results['distances'][0][i]
```
- Filters results by similarity threshold (< 0.8)
- Calculates relevance score (1 - distance)
- Higher score = more relevant

**Step 4: Ranking and Selection**
```python
formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
return formatted_results[:3]
```
- Sorts by relevance score (highest first)
- Returns top 3 most relevant results

## 🤖 AI Response Generation

### Multi-Model Architecture

**Primary: Ollama (Local)**
```python
payload = {
    "model": "llama3.2:3b",
    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    "stream": False,
    "options": {"temperature": 0.3, "max_tokens": 400}
}
```

**Secondary: OpenRouter API (Enhancement)**
```python
payload = {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    "max_tokens": 800,
    "temperature": 0.3
}
```

**Tertiary: OpenAI API (Fallback)**
```python
payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    "max_tokens": 400,
    "temperature": 0.3
}
```

## 🌍 Language Processing

### Intelligent Language Conversion

**Language Detection**
```python
def is_arabic_text(self, text: str) -> bool:
    arabic_chars = 0
    total_chars = 0
    
    for char in text:
        if char.isalpha():
            total_chars += 1
            if '\u0600' <= char <= '\u06FF':  # Arabic Unicode range
                arabic_chars += 1
    
    return total_chars > 0 and (arabic_chars / total_chars) > 0.3
```

**Query Translation**
- Detects document language (Arabic/English)
- Translates query to match document language
- Uses OpenRouter API for accurate translation
- Responds in original question language

## 📦 Required Packages

### Core Dependencies
```txt
streamlit>=1.28.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
langchain>=0.1.0
langchain-community>=0.0.10
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
openai>=1.0.0
```

### Optional Dependencies
```txt
torch>=2.0.0          # For local models
transformers>=4.30.0  # For Hugging Face models
accelerate>=0.20.0    # For model optimization
bitsandbytes>=0.41.0  # For quantization
```

## 🚀 Performance Optimizations

### Response Speed
- **Reduced max_tokens**: 800 (faster generation)
- **Lower temperature**: 0.3 (more focused responses)
- **Shorter context**: 1000 characters (faster processing)
- **Efficient chunking**: 1000 char chunks with 200 overlap

### Memory Management
- **Quantization**: 4-bit quantization for local models
- **Chunking**: Prevents memory overflow with large documents
- **Session state**: Efficient chat history management
- **Vector caching**: Persistent ChromaDB storage

### Accuracy Improvements
- **Relevance scoring**: Filters results by similarity threshold
- **Context ranking**: Sorts by relevance score
- **Language conversion**: Matches query language to document language
- **Multi-model fallback**: Ensures response availability

## 🔧 Configuration

### Environment Variables
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
API_KEY=your_openai_api_key_here
```

### Configurable Parameters
```python
CHUNK_SIZE = 1000              # Document chunk size
CHUNK_OVERLAP = 200            # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Embedding model
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"  # Enhancement model
```

## 📊 System Flow

1. **Document Upload** → PDF/Website Processing → Text Extraction → Chunking → Vector Storage
2. **Question Asked** → Language Detection → Query Translation → Vector Search → Context Retrieval
3. **Answer Generation** → RAG Response → OpenRouter Enhancement → Language-Specific Response
4. **Chat History** → Session Storage → Tile Display → Language Categorization

This modular architecture ensures maintainability, scalability, and easy debugging while providing a robust RAG system for multilingual document Q&A.
