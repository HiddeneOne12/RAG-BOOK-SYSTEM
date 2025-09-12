import streamlit as st
import os
import time
from pathlib import Path
import base64
from rag_system import RAGSystem
from config import PDF_FOLDER

# Page configuration
st.set_page_config(
    page_title="📚 RAG Book Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .question-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .answer-card {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .source-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def load_rag_system():
    """Load the RAG system"""
    try:
        with st.spinner("🔄 Initializing RAG system..."):
            rag_system = RAGSystem()
            st.session_state.rag_system = rag_system
            st.success("✅ RAG system initialized successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")
        return False

def load_model():
    """Load the Gemma model"""
    try:
        with st.spinner("🔄 Loading Gemma 1B model (this may take a few minutes)..."):
            st.session_state.rag_system.load_model()
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return False

def process_pdfs():
    """Process PDFs in the folder"""
    try:
        with st.spinner("🔄 Processing PDFs..."):
            st.session_state.rag_system.process_pdf_folder()
            st.session_state.documents_processed = True
            st.success("✅ PDFs processed successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {str(e)}")
        return False

def display_pdf_upload():
    """Display PDF upload section"""
    st.markdown("### 📁 PDF Management")
    
    # Check if PDF folder exists and show files
    if os.path.exists(PDF_FOLDER):
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
        if pdf_files:
            st.markdown("**Current PDFs in folder:**")
            for pdf_file in pdf_files:
                st.markdown(f"📄 {pdf_file}")
        else:
            st.warning("No PDF files found in the pdfs folder. Please add some PDF files.")
    else:
        st.warning("PDF folder not found. Please create a 'pdfs' folder and add PDF files.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=['pdf'],
        help="Upload a PDF file to add to your knowledge base"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File {uploaded_file.name} uploaded successfully!")
        
        # Process the uploaded file
        if st.button("🔄 Process Uploaded PDF"):
            try:
                with st.spinner("Processing uploaded PDF..."):
                    documents = st.session_state.rag_system.process_pdf(file_path)
                    st.session_state.rag_system.add_documents_to_collection(documents, uploaded_file.name)
                    st.success("✅ PDF processed and added to knowledge base!")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")

def display_question_interface():
    """Display the question interface"""
    st.markdown("### 💬 Ask Questions")
    
    # Question input
    question = st.text_area(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic discussed in the book?",
        height=100,
        help="Ask contextual or analytical questions about your uploaded documents"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and question:
        if not st.session_state.model_loaded:
            st.error("❌ Please load the model first!")
            return
        
        if not st.session_state.documents_processed:
            st.error("❌ Please process PDFs first!")
            return
        
        # Process question
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.rag_system.ask_question(question)
                
                # Debug information
                st.write("🔍 Debug Info:")
                st.write(f"Answer length: {len(result['answer']) if result['answer'] else 0}")
                st.write(f"Sources: {result['sources']}")
                st.write(f"Context docs: {len(result['context_documents'])}")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'timestamp': time.time()
                })
            except Exception as e:
                st.error(f"Error processing question: {e}")
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': f"Error: {e}",
                    'sources': [],
                    'timestamp': time.time()
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💭 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                if chat['sources']:
                    st.markdown(f"**Sources:** {', '.join(set(chat['sources']))}")

def display_metrics():
    """Display system metrics"""
    st.markdown("### 📊 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Status", "✅ Loaded" if st.session_state.model_loaded else "❌ Not Loaded")
    
    with col2:
        st.metric("Documents Processed", "✅ Yes" if st.session_state.documents_processed else "❌ No")
    
    with col3:
        if st.session_state.rag_system and st.session_state.rag_system.collection:
            try:
                count = st.session_state.rag_system.collection.count()
                st.metric("Document Chunks", count)
            except:
                st.metric("Document Chunks", "N/A")
        else:
            st.metric("Document Chunks", "N/A")
    
    with col4:
        st.metric("Chat Messages", len(st.session_state.chat_history))
    
    # AI Enhancement Status
    st.markdown("### 🤖 AI Enhancement Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check OpenRouter status
        from config import OPENROUTER_API_KEY
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
            st.success("✅ OpenRouter API Ready")
        else:
            st.warning("⚠️ OpenRouter API Not Configured")
    
    with col2:
        # Check Ollama status
        if st.session_state.rag_system:
            if st.session_state.rag_system._check_ollama_available():
                st.success("✅ Ollama Available")
            else:
                st.info("ℹ️ Ollama Not Running")
        else:
            st.info("ℹ️ Ollama Status Unknown")
    
    with col3:
        # Check API fallback status
        from config import API_FALLBACK_ENABLED, API_KEY
        if API_FALLBACK_ENABLED and API_KEY and API_KEY != "your_openai_api_key_here":
            st.success("✅ OpenAI Fallback Ready")
        else:
            st.info("ℹ️ OpenAI Fallback Not Configured")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 RAG Book Assistant</h1>
        <p>Ask questions about your PDF documents using AI-powered retrieval and generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Setup")
        
        # Initialize RAG system
        if st.button("🔧 Initialize RAG System"):
            load_rag_system()
        
        # Load model
        if st.button("🤖 Load Gemma Model") and st.session_state.rag_system:
            load_model()
        
        # Process PDFs
        if st.button("📚 Process PDFs") and st.session_state.rag_system:
            process_pdfs()
        
        st.markdown("---")
        
        # Display metrics
        display_metrics()
        
        st.markdown("---")
        
        # PDF management
        display_pdf_upload()
    
    # Main content
    if not st.session_state.rag_system:
        st.info("👈 Please initialize the RAG system from the sidebar to get started.")
        return
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the Gemma model from the sidebar to ask questions.")
        return
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please process PDFs from the sidebar to enable question answering.")
        return
    
    # Question interface
    display_question_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using Streamlit, ChromaDB, and Gemma 1B</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
