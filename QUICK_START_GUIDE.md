# 🚀 Quick Start Guide - RAG Book Assistant

## ⚡ Super Quick Start (Recommended)

```bash
# 1. Navigate to your project directory
cd /Users/apple/python-projects/rag_system

# 2. Run the quick start script
python quick_start.py
```

That's it! The script will:
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Create a sample PDF for testing
- ✅ Test the system
- ✅ Launch the web application

## 📋 Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create directories
mkdir pdfs chroma_db

# 3. Create sample PDF (optional)
python create_sample_pdf.py

# 4. Test the system
python test_system.py

# 5. Run the application
python run.py
# OR
streamlit run app.py
```

## 🎯 How to Use

### Step 1: Setup (One-time)
1. Open your browser to `http://localhost:8501`
2. In the sidebar, click "🔧 Initialize RAG System"
3. Click "🤖 Load Gemma Model" (takes 2-3 minutes)
4. Click "📚 Process PDFs"

### Step 2: Add Your PDFs
- Place PDF files in the `pdfs` folder
- Or use the upload feature in the sidebar

### Step 3: Ask Questions
- Type your question in the text area
- Click "🔍 Ask Question"
- Get AI-powered answers!

## 💡 Example Questions

**Contextual Questions:**
- "What is the main topic of chapter 3?"
- "What does the author say about machine learning?"
- "Can you summarize the key points?"

**Analytical Questions:**
- "Compare the arguments in section 1 and 2"
- "What are the pros and cons mentioned?"
- "How does this relate to current technology trends?"

## 🛠️ Troubleshooting

### Common Issues:

1. **"Model not loaded" error**
   - Wait for the model to fully load (2-3 minutes)
   - Check if you have enough RAM (4GB+ recommended)

2. **"No PDFs found" error**
   - Add PDF files to the `pdfs` folder
   - Make sure files have `.pdf` extension

3. **Memory issues**
   - Close other applications
   - Restart the application
   - The system uses 4-bit quantization to save memory

4. **Slow responses**
   - First question may take longer
   - Subsequent questions will be faster
   - Consider using a GPU if available

## 📁 Project Structure

```
rag_system/
├── app.py                    # Main Streamlit app
├── rag_system.py            # Core RAG system
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── quick_start.py           # Quick setup script
├── run.py                   # Launcher script
├── setup.py                 # Manual setup script
├── test_system.py           # System tests
├── create_sample_pdf.py     # Sample PDF creator
├── README.md                # Full documentation
├── QUICK_START_GUIDE.md     # This file
├── pdfs/                    # Your PDF files go here
└── chroma_db/              # Vector database (auto-created)
```

## 🎉 Features

- ✅ **PDF Processing**: Automatically chunks and indexes PDFs
- ✅ **Vector Search**: ChromaDB for fast similarity search
- ✅ **AI Answers**: Gemma 1B model for intelligent responses
- ✅ **Beautiful UI**: Modern, responsive interface
- ✅ **Memory Optimized**: 4-bit quantization for lower RAM usage
- ✅ **Contextual & Analytical**: Handles complex questions
- ✅ **Chat History**: Keeps track of your conversations
- ✅ **Source Attribution**: Shows which documents were used

## 🔧 Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Model settings
- Directory paths

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_system.py` to diagnose problems
3. Check the console output for error messages

## 🎯 Next Steps

1. Add your own PDF documents
2. Experiment with different types of questions
3. Customize the configuration if needed
4. Enjoy your AI-powered document assistant!

---

**Happy Reading! 📚✨**
