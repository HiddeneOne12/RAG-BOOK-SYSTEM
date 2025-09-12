# 📚 RAG Book Assistant with OpenRouter AI Enhancement

A powerful Retrieval-Augmented Generation (RAG) system that allows you to ask questions about your PDF documents using AI. Built with ChromaDB for vector storage and enhanced with OpenRouter API for coherent, user-friendly responses.

## ✨ Features

- **PDF Processing**: Automatically processes PDF files and converts them into searchable chunks
- **Vector Search**: Uses ChromaDB for efficient similarity search
- **AI-Enhanced Responses**: 
  - **OpenRouter API** (primary) - Provides coherent, well-structured explanations
  - **Ollama with Llama 3.2 3B** (secondary) - Local AI processing
  - **OpenAI API fallback** (tertiary) - Reliable cloud-based responses
  - **Local fallback** (always available) - Basic structured responses
- **Beautiful UI**: Modern, responsive interface built with Streamlit
- **Memory Optimized**: Uses quantization to reduce memory usage
- **Contextual & Analytical Questions**: Handles both simple and complex queries
- **Smart Fallback System**: Automatically switches between AI models for best performance
- **Enhanced Coherence**: OpenRouter AI makes responses more user-friendly and comprehensive

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- At least 4GB RAM (recommended 8GB)
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone or download this repository**
   ```bash
   cd /Users/apple/python-projects/rag_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup API configuration (recommended for best performance)**
   ```bash
   cp env_template.txt .env
   ```
   Edit the `.env` file to add your API keys:
   - **OpenRouter API key** (primary) - Get from https://openrouter.ai/keys
   - **OpenAI API key** (fallback) - Get from https://platform.openai.com/api-keys

4. **Create PDF folder**
   ```bash
   mkdir pdfs
   ```

5. **Add your PDF files**
   - Place your PDF files in the `pdfs` folder
   - The system will automatically process them

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Follow the setup steps in the sidebar

## 📖 How to Use

### Step 1: Setup
1. Click "🔧 Initialize RAG System" in the sidebar
2. Click "🤖 Load Gemma Model" (this may take a few minutes)
3. Click "📚 Process PDFs" to process your documents
 
### Step 2: Ask Questions
1. Type your question in the text area
2. Click "🔍 Ask Question"
3. Get AI-powered answers based on your documents

### Example Questions
- **Contextual**: "What is the main topic of chapter 3?"
- **Analytical**: "Compare the arguments presented in the first and second sections"
- **Specific**: "What does the author say about machine learning?"
- **Summary**: "Can you summarize the key points from this document?"

## 🛠️ Configuration

You can modify settings in `config.py`:

```python
CHUNK_SIZE = 1000          # Size of text chunks
CHUNK_OVERLAP = 200        # Overlap between chunks
MODEL_NAME = "google/gemma-1.1-1b-it"  # Model to use
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
```

## 📁 Project Structure

```
rag_system/
├── app.py                 # Main Streamlit application
├── rag_system.py         # Core RAG system implementation
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── pdfs/                # Folder for PDF files
└── chroma_db/           # ChromaDB storage (created automatically)
```

## 🔧 Technical Details

### RAG System Components
- **Document Processing**: PyPDF2 for PDF extraction
- **Text Chunking**: LangChain's RecursiveCharacterTextSplitter
- **Vector Storage**: ChromaDB with cosine similarity
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **AI Enhancement**: OpenRouter API with Llama 3.1 8B (primary)
- **Text Generation**: Google Gemma 1B with 4-bit quantization (fallback)
- **Response Enhancement**: Two-stage process - RAG retrieval + AI enhancement

### Memory Optimization
- Uses BitsAndBytesConfig for 4-bit quantization
- Optimized chunk sizes for efficient processing
- Persistent ChromaDB storage to avoid reprocessing

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure you have enough RAM (at least 4GB)
   - Try restarting the application
   - Check if you have the required dependencies

2. **PDF Processing Error**
   - Ensure PDF files are not corrupted
   - Check if files are in the `pdfs` folder
   - Verify file permissions

3. **Memory Issues**
   - Close other applications to free up RAM
   - Reduce chunk size in config.py
   - Use CPU instead of GPU if available

### Performance Tips
- Use smaller chunk sizes for better precision
- Process PDFs in batches if you have many files
- Restart the application periodically to clear memory

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Google Gemma](https://huggingface.co/google/gemma-1.1-1b-it) for text generation
- [Streamlit](https://streamlit.io/) for the beautiful UI
- [LangChain](https://langchain.com/) for document processing
