import streamlit as st
import os
import json
from datetime import datetime
from content_manager import ContentManager

# Initialize content manager
@st.cache_resource
def get_content_manager():
    return ContentManager()

# Chat history functions
def load_chat_history():
    """Load chat history from session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

def save_chat_history(question, answer, language):
    """Save question and answer to chat history"""
    chat_entry = {
        'question': question,
        'answer': answer,
        'language': language,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.chat_history.insert(0, chat_entry)  # Add to beginning
    
    # Keep only last 20 conversations
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[:20]

def clear_chat_history():
    """Clear all chat history"""
    st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="AI Knowledge Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Knowledge Assistant")
    st.markdown("Upload documents or add website content, then ask questions about them.")
    
    # Initialize content manager
    content_manager = get_content_manager()
    
    # Sidebar for content management
    with st.sidebar:
        st.header("📚 Add Content")
        
        # PDF Upload with automatic processing
        st.subheader("📄 Upload PDF")
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file", 
            type=['pdf'], 
            help="PDF will be automatically processed when selected"
        )
        
        if uploaded_pdf is not None:
            # Auto-process PDF
            with st.spinner("Processing PDF..."):
                temp_path = f"temp_{uploaded_pdf.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                success = content_manager.add_pdf(temp_path)
                os.remove(temp_path)  # Clean up temp file
                
                if success:
                    st.success(f"✅ {uploaded_pdf.name} added successfully!")
                else:
                    st.error(f"❌ Failed to process {uploaded_pdf.name}")
        
        # Website URL with automatic processing
        st.subheader("🌐 Add Website")
        website_url = st.text_input(
            "Enter Website URL", 
            placeholder="https://example.com",
            help="Website will be automatically processed when you press Enter"
        )
        
        if website_url and st.button("Add Website", type="primary"):
            with st.spinner("Processing website..."):
                success = content_manager.add_website(website_url)
                
                if success:
                    st.success(f"✅ Website added successfully!")
                else:
                    st.error(f"❌ Failed to process website")
        
        # Process existing PDFs
        st.subheader("📁 Process Existing PDFs")
        if st.button("Load PDFs from ./pdfs folder"):
            with st.spinner("Processing PDFs..."):
                result = content_manager.process_pdf_folder("./pdfs")
                
                if result["success"]:
                    st.success(f"✅ Processed {len(result['processed'])} PDFs!")
                else:
                    st.error(f"❌ {result['message']}")
        
        # Show processed sources
        st.subheader("📋 Knowledge Base")
        sources = content_manager.get_processed_sources()
        if sources:
            for source in sources:
                st.write(f"• {source}")
        else:
            st.write("No content added yet")
    
    # Main content area
    st.header("💬 Ask Questions")
    
    # Load chat history
    chat_history = load_chat_history()
    
    # Question input
    question = st.text_area(
        "What would you like to know?",
        placeholder="Ask any question about your uploaded content...",
        height=120,
        help="Ask questions about the documents and websites you've added"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Get Answer", type="primary", use_container_width=True):
            if question:
                with st.spinner("Thinking..."):
                    result = content_manager.ask_question(question)
                    
                    # Detect language of question
                    is_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
                    language = "Arabic" if is_arabic else "English"
                    
                    # Save to chat history
                    save_chat_history(question, result['answer'], language)
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(result['answer'])
                    
                    # Display sources if available
                    if result['sources']:
                        with st.expander("📚 Sources"):
                            for source in result['sources']:
                                st.write(f"• {source}")
            else:
                st.warning("Please enter a question")
    
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            clear_chat_history()
            st.rerun()
    
    # Display chat history
    if chat_history:
        st.markdown("---")
        st.subheader("📝 Recent Questions")
        
        # Create columns for chat tiles
        cols = st.columns(2)
        
        for i, chat in enumerate(chat_history):
            with cols[i % 2]:
                # Determine tile color based on language
                if chat['language'] == "Arabic":
                    tile_color = "#e3f2fd"  # Light blue for Arabic
                    lang_icon = "🇸🇦"
                else:
                    tile_color = "#f3e5f5"  # Light purple for English
                    lang_icon = "🇺🇸"
                
                # Create tile
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background-color: {tile_color};
                        padding: 15px;
                        border-radius: 10px;
                        margin: 10px 0;
                        border-left: 4px solid {'#2196f3' if chat['language'] == 'Arabic' else '#9c27b0'};
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="font-weight: bold; color: #333;">{lang_icon} {chat['language']}</span>
                            <span style="font-size: 0.8em; color: #666;">{chat['timestamp']}</span>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <strong>Q:</strong> {chat['question'][:100]}{'...' if len(chat['question']) > 100 else ''}
                        </div>
                        <div>
                            <strong>A:</strong> {chat['answer'][:150]}{'...' if len(chat['answer']) > 150 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📄 PDF Analysis**
        - Upload and analyze PDF documents
        - Automatic content extraction
        - Intelligent text chunking
        """)
    
    with col2:
        st.markdown("""
        **🌐 Website Analysis**
        - Extract content from any website
        - Clean text processing
        - Real-time content analysis
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI-Powered Answers**
        - Advanced RAG technology
        - Precise and contextual responses
        - Multiple AI model support
        """)

if __name__ == "__main__":
    main()