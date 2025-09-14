# 🤖 RAG System with Website & PDF Support

A powerful Retrieval-Augmented Generation (RAG) system that supports both PDF documents and website content analysis, enhanced with OpenRouter AI for better response quality.

## ✨ Features

### 📄 **PDF Support**
- Upload and process PDF documents
- Automatic text extraction and chunking
- Process entire folders of PDFs
- Support for multiple PDF formats

### 🌐 **Website Support**
- Analyze content from any accessible website
- Clean text extraction with HTML parsing
- Automatic content chunking
- URL validation and error handling

### 🤖 **AI Enhancement**
- **RAG + OpenRouter Integration**: First generates RAG response, then enhances with OpenRouter AI
- **Multiple Fallback Options**: Ollama → OpenAI → Local Fallback
- **Coherent Responses**: Enhanced explanations and user-friendly formatting
- **Debug Logging**: Track the entire process with detailed logs

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd rag_system

# Create virtual environment
python3 -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configuration**
Create a `.env` file:
```bash
# OpenRouter API (for enhanced responses)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
OPENROUTER_TIMEOUT=30

# OpenAI API (fallback)
OPENAI_API_KEY=your_openai_api_key_here
API_FALLBACK_ENABLED=true

# Ollama (local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### 3. **Run the Application**
```bash
# Start the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
rag_system/
├── rag_system.py          # Core RAG system (300 lines max)
├── website_processor.py   # Website content processing
├── content_manager.py     # Unified content management
├── app.py                # Streamlit web interface
├── config.py             # Configuration settings
├── test_system.py        # Test script
├── requirements.txt      # Dependencies
├── README.md            # This file
└── pdfs/                # PDF documents folder
```

## 🔧 Usage

### **Web Interface**
1. Open `http://localhost:8501` in your browser
2. **Add Content**:
   - Upload PDF files using the sidebar
   - Enter website URLs to analyze
   - Process entire PDF folders
3. **Ask Questions**: Type questions about your content
4. **View Results**: Get enhanced, coherent answers

### **Programmatic Usage**
```python
from content_manager import ContentManager

# Initialize
cm = ContentManager()

# Add PDF
cm.add_pdf("document.pdf")

# Add website
cm.add_website("https://example.com")

# Ask questions
result = cm.ask_question("What is this about?")
print(result['answer'])
```

## 🧪 Testing

Run the test script:
```bash
python test_system.py
```

This will test:
- PDF processing
- Website content extraction
- Question answering functionality

## 🔄 How It Works

### **1. Content Processing**
- **PDFs**: Extract text → Chunk → Generate embeddings → Store in ChromaDB
- **Websites**: Scrape content → Clean HTML → Chunk → Generate embeddings → Store in ChromaDB

### **2. Question Answering**
1. **Retrieval**: Find relevant document chunks using semantic search
2. **RAG Generation**: Generate initial response using Ollama/OpenAI/Local fallback
3. **OpenRouter Enhancement**: Enhance response for better coherence and explanation
4. **Return**: Deliver enhanced, user-friendly answer

### **3. Fallback System**
- **Primary**: Ollama (local, fast)
- **Secondary**: OpenAI API (high quality)
- **Tertiary**: Local fallback (always works)

## 🛠️ Configuration Options

### **ChromaDB Settings**
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Model**: `all-MiniLM-L6-v2`

### **API Settings**
- **OpenRouter**: Enhanced response generation
- **OpenAI**: Fallback option
- **Ollama**: Local processing

## 📊 Performance

- **PDF Processing**: ~2-5 seconds per document
- **Website Processing**: ~3-8 seconds per URL
- **Question Answering**: ~5-15 seconds (depending on API)
- **Memory Usage**: Optimized with 4-bit quantization

## 🔒 Security

- API keys stored in `.env` file
- No sensitive data logged
- Local processing options available
- URL validation for websites

## 🐛 Troubleshooting

### **Common Issues**
1. **OpenRouter API Error**: Check API key and model name
2. **Ollama Not Available**: Install and start Ollama service
3. **PDF Processing Failed**: Check file format and permissions
4. **Website Access Denied**: Some sites block automated access

### **Debug Mode**
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for vector storage
- **OpenRouter** for AI enhancement
- **Streamlit** for the web interface
- **LangChain** for document processing
- **Hugging Face** for transformer models

---

**Made with ❤️ for the AI community**