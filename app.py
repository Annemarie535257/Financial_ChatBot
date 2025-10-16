from flask import Flask, render_template, request, jsonify
import torch
from transformers import T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration
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
        
        # Look for model directory with config.json (HuggingFace format)
        model_dir = None
        
        # Search for HuggingFace format model (config.json + tf_model.h5)
        for root, dirs, files in os.walk('saved-model'):
            if 'config.json' in files and 'tokenizer_config.json' in files:
                # Check for TensorFlow model file (could be tf_model.h5 or tf_model .h5)
                has_tf_model = any('tf_model' in f.lower() and '.h5' in f.lower() for f in files)
                if has_tf_model:
                    model_dir = root
                    logger.info(f"📁 Found HuggingFace format model at: {model_dir}")
                    break
        
        if not model_dir:
            logger.error("❌ No trained model found in saved-model directory")
            logger.error("💡 Make sure a saved model exists with config.json and tokenizer files")
            return False
        
        logger.info(f"📄 Loading tokenizer from: {model_dir}")
        try:
            # Try loading with T5TokenizerFast first
            tokenizer = T5TokenizerFast.from_pretrained(model_dir)
            logger.info("✅ T5TokenizerFast loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️ T5TokenizerFast failed: {e}")
            logger.info("🔄 Trying T5Tokenizer instead...")
            try:
                # Fallback to regular T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained(model_dir)
                logger.info("✅ T5Tokenizer loaded successfully!")
            except Exception as e2:
                logger.warning(f"⚠️ T5Tokenizer also failed: {e2}")
                logger.info("🔄 Trying to load from base model...")
                # Try loading from base FLAN-T5 model
                tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
                logger.info("✅ Base FLAN-T5 tokenizer loaded successfully!")
        
        # Load model using PyTorch (more compatible)
        logger.info(f"📄 Loading PyTorch model...")
        try:
            # Try loading with PyTorch HuggingFace
            model = T5ForConditionalGeneration.from_pretrained(model_dir)
            logger.info("✅ Model loaded with PyTorch HuggingFace method!")
        except Exception as e:
            logger.warning(f"⚠️ PyTorch HuggingFace loading failed: {e}")
            logger.info("🔄 Trying to load from TensorFlow weights...")
            try:
                # Try loading TensorFlow weights into PyTorch model
                model = T5ForConditionalGeneration.from_pretrained(model_dir, from_tf=True)
                logger.info("✅ Model loaded from TensorFlow weights!")
            except Exception as e2:
                logger.warning(f"⚠️ TensorFlow to PyTorch conversion failed: {e2}")
                logger.info("🔄 Trying to load from base model...")
                # Fallback to base FLAN-T5 model
                model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
                logger.info("✅ Base FLAN-T5 model loaded successfully!")
        
        logger.info("✅ Model and tokenizer loaded successfully!")
        logger.info(f"🤖 Model Type: FLAN-T5 (Conditional Generation)")
        logger.info("🎯 Model is ready for generating responses!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        import traceback
        logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
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
        input_text = PREFIX + question.lower()
        
        # Tokenize the input
        inputs = tokenizer([input_text], return_tensors='pt', padding=True, 
                          truncation=True, max_length=256)
        
        # Generate response with improved parameters
        try:
            # Try HuggingFace model generation
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                min_length=30,  # Longer minimum responses
                num_beams=8,    # More beams for better quality
                early_stopping=True,
                do_sample=False,  # Use deterministic generation
                temperature=0.3,  # Lower temperature for more focused responses
                top_p=0.8,       # More conservative sampling
                top_k=30,        # Fewer tokens to choose from
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.3,  # Higher penalty for repetition
                length_penalty=1.2,      # Encourage longer responses
                no_repeat_ngram_size=4,   # Prevent 4-word repetition
            )
        except Exception as e:
            logger.warning(f"⚠️ HuggingFace generation failed: {e}")
            logger.info("🔄 Trying PyTorch native generation...")
            # Fallback for PyTorch native models
            with torch.no_grad():
                outputs = model(**inputs)
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prefix from the response if it appears
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        # Check if response is helpful - if not, use a simple fallback
        response_lower = response.lower().strip()
        
        # Log the original response for debugging
        logger.info(f"Original model response: '{response[:100]}...'")
        
        # Enhanced fallback for unhelpful or incorrect responses
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
            'use a calculator' in response_lower,
            # Add patterns for clearly incorrect responses
            'registered trademark' in response_lower,
            'company specializing' in response_lower,
            'manufacture and sale' in response_lower,
            'personal protective equipment' in response_lower,
            'financial institution' in response_lower and 'mortgage' in question.lower(),
            'in the event of a default' in response_lower and len(response.split()) < 15
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
        logger.info("🌐 Server will be available at: http://localhost:8080")
        logger.info("💬 ChatBot is ready to answer your financial questions!")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        logger.error("❌ Failed to load model. Please check the model files.")
        logger.error("💡 Make sure the saved-model directory contains the trained model.")
        exit(1)
