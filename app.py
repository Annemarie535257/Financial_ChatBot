from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from transformers import T5TokenizerFast
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info("🚀 Loading Financial ChatBot model...")
        
        # Find the saved model directory (look for Hugging Face format)
        model_dir = None
        for root, dirs, files in os.walk('saved_models'):
            if 'config.json' in files:  # Look for HF format
                model_dir = root
                break
        
        if not model_dir:
            logger.error("❌ No trained model found in saved_models directory")
            logger.error("💡 Make sure to run the updated notebook with compatible model saving")
            return False
            
        logger.info(f"📁 Found saved model at: {model_dir}")
        
        # Load model and tokenizer using Hugging Face format
        model = TFT5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5TokenizerFast.from_pretrained(model_dir)
        
        logger.info("✅ Model and tokenizer loaded successfully!")
        logger.info("🎯 Using fully trained model - responses should be financial-focused!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.error("💡 Make sure to retrain your model using the updated notebook")
        return False

def generate_response(question, max_new_tokens=128, temperature=0.7):
    """Generate response using the trained model"""
    try:
        if model is None or tokenizer is None:
            logger.error("Model or tokenizer not loaded")
            return "I apologize, but the AI model is not currently available. Please try again later."
        
        logger.info(f"🤖 Generating response using trained model for: {question[:50]}...")
        
        # Format the input with the same prefix used in training (from your notebook)
        PREFIX = 'answer the question: '
        input_text = PREFIX + question
        
        # Tokenize the input
        inputs = tokenizer([input_text], return_tensors='tf', padding=True, 
                          truncation=True, max_length=256)
        
        # Generate response with improved parameters
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            min_length=20,  # Ensure minimum response length
            num_beams=6,    # More beams for better quality
            early_stopping=True,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prefix from the response if it appears
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        # Check if response is helpful - if not, use a simple fallback
        response_lower = response.lower().strip()
        
        # Log the original response for debugging
        logger.info(f"Original model response: '{response[:100]}...'")
        
        # Simple fallback for unhelpful responses
        unhelpful_patterns = [
            len(response.strip()) < 30,
            'would you like to make' in response_lower,
            'i want to make a monthly' in response_lower,
            response_lower.count('make') > 2,
            len(response.split()) < 10,
            response.endswith('...'),
            response in ['no', 'yes', 'ok', 'apply online'],
            'is a form of payment' in response_lower,
            'can be used to purchase' in response_lower,
            'you will need to apply' in response_lower,
            'you can use a debit card' in response_lower,
            'use a calculator' in response_lower
        ]
        
        if any(unhelpful_patterns):
            logger.info("🔄 Using fallback response for better quality")
            response = "I understand you have a financial question. While I can provide general information about loans, mortgages, and financial topics, I recommend consulting with a qualified financial advisor for specific advice tailored to your situation. For immediate assistance, you can contact your loan servicer directly or visit your local bank or credit union."
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

@app.route('/')
def index():
    """Serve the main landing page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chatbot interaction"""
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        # Generate response
        response = generate_response(question)
        
        return jsonify({
            'success': True,
            'response': response,
            'question': question
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'message': 'Financial ChatBot API is running'
    })

if __name__ == '__main__':
    # Load model on startup
    logger.info("🚀 Starting Financial ChatBot API...")
    
    if load_model():
        logger.info("✅ Model loaded successfully! Starting Flask app...")
        logger.info("🌐 Server will be available at: http://localhost:5000")
        logger.info("💬 ChatBot is ready to answer your financial questions!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to load model. Please check the model files.")
        logger.error("💡 Make sure the saved_models directory contains the trained model.")
        exit(1)
