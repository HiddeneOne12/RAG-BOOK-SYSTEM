#!/usr/bin/env python3
"""
Launcher script for RAG Book Assistant
"""

import subprocess
import sys
import os
from pathlib import Path

def check_setup():
    """Check if the system is properly set up"""
    required_files = ['app.py', 'rag_system.py', 'config.py', 'requirements.txt']
    required_dirs = ['pdfs']
    
    # Check files
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    # Check directories
    for directory in required_dirs:
        if not Path(directory).exists():
            print(f"❌ Missing required directory: {directory}")
            return False
    
    print("✅ All required files and directories found")
    return True

def main():
    """Main launcher function"""
    print("🚀 Starting RAG Book Assistant...")
    print("=" * 50)
    
    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please run: python setup.py")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n🌐 Starting web application...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n💡 Tip: Add PDF files to the 'pdfs' folder before asking questions")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")

if __name__ == "__main__":
    main()
