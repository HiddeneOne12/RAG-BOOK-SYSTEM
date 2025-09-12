#!/usr/bin/env python3
"""
Setup script for OpenRouter integration
"""

import os
import shutil

def setup_openrouter():
    """Setup OpenRouter configuration"""
    print("🔧 Setting up OpenRouter Integration...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        if os.path.exists('env_template.txt'):
            shutil.copy('env_template.txt', '.env')
            print("✅ Created .env file from template")
        else:
            print("❌ env_template.txt not found")
            return False
    else:
        print("✅ .env file already exists")
    
    # Check for OpenRouter API key
    with open('.env', 'r') as f:
        content = f.read()
    
    if 'your_openrouter_api_key_here' in content:
        print("\n🔑 OpenRouter API Key Setup Required:")
        print("1. Get an API key from: https://openrouter.ai/keys")
        print("2. Edit the .env file and replace 'your_openrouter_api_key_here' with your actual API key")
        print("\n📝 Example .env configuration:")
        print("OPENROUTER_API_KEY=sk-or-your-actual-api-key-here")
        print("OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free")
        
        return False
    else:
        print("✅ OpenRouter API key appears to be configured")
        return True

def test_openrouter_config():
    """Test OpenRouter configuration"""
    print("\n🧪 Testing OpenRouter Configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from config import OPENROUTER_API_KEY, OPENROUTER_MODEL
        
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
            print("✅ OpenRouter API key is configured")
            print(f"✅ Using model: {OPENROUTER_MODEL}")
            return True
        else:
            print("❌ OpenRouter API key is not properly configured")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenRouter configuration: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 RAG System OpenRouter Setup")
    print("=" * 50)
    
    # Setup configuration
    if setup_openrouter():
        # Test configuration
        if test_openrouter_config():
            print("\n🎉 OpenRouter setup complete!")
            print("Your RAG system will now use:")
            print("1. OpenRouter API (primary) - for enhanced, coherent responses")
            print("2. Ollama (secondary) - if OpenRouter is unavailable")
            print("3. OpenAI API (tertiary) - if both above fail")
            print("4. Local fallback (always available) - basic responses")
        else:
            print("\n⚠️ OpenRouter setup incomplete")
            print("Please configure your API key in the .env file")
    else:
        print("\n⚠️ Please complete the OpenRouter API key setup")
    
    print("\n📚 Your RAG system is ready to use!")
    print("Run: streamlit run app.py")

if __name__ == "__main__":
    main()
