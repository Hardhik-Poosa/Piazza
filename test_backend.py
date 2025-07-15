#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing backend imports...")
    
    # Test basic imports
    from flask import Flask
    print("✓ Flask imported successfully")
    
    import torch
    print("✓ PyTorch imported successfully")
    
    from PIL import Image
    print("✓ PIL imported successfully")
    
    # Test optional imports
    try:
        from flask_cors import CORS
        print("✓ flask-cors imported successfully")
    except ImportError:
        print("⚠ flask-cors not available")
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        print("✓ transformers imported successfully")
    except ImportError:
        print("⚠ transformers not available")
    
    try:
        from olmocr import render_pdf_to_base64png, get_anchor_text, build_finetuning_prompt
        print("✓ olmocr imported successfully")
    except ImportError:
        print("⚠ olmocr not available")
    
    # Test Flask app creation
    print("\nTesting Flask app creation...")
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "Test backend is running!"
    
    print("✓ Flask app created successfully")
    
    # Test if we can start the server
    print("\nStarting test server on port 5001...")
    app.run(debug=True, port=5001, host='127.0.0.1')
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
