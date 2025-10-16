#!/usr/bin/env python3
"""
Financial ChatBot Web Application
Run this script to start the Flask web server with the AI chatbot
"""

import os
import sys
from app import app, load_model

def main():
    """Main function to run the Flask application"""
    print("🤖 Financial ChatBot Web Application")
    print("=" * 50)
    
    # Check if model files exist
    model_found = False
    for root, dirs, files in os.walk('saved-model'):
        if 'config.json' in files and 'tokenizer_config.json' in files:
            has_tf_model = any('tf_model' in f.lower() and '.h5' in f.lower() for f in files)
            if has_tf_model:
                model_found = True
                break
    
    if not model_found:
        print("⚠️  Warning: No trained model found in saved-model directory")
        print("   The chatbot will run in demo mode")
        print("   To train a model, run the Jupyter notebook first")
    else:
        print("✅ Model files found (HuggingFace format)")
    
    print("\n🚀 Starting Flask server...")
    print("📍 Server will be available at: http://localhost:8080")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Load model if available
        if model_found:
            if load_model():
                print("✅ Model loaded successfully")
            else:
                print("❌ Failed to load model, running in demo mode")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=8080)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
