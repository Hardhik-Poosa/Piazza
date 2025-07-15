import os
import json
import re
import uuid
import tempfile
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch

# Try to import flask-cors, but continue if not available
try:
    from flask_cors import CORS
    has_cors = True
except ImportError:
    has_cors = False
    print("Warning: flask-cors not installed. CORS support disabled.")

# Try to import transformers and olmocr, but use mock implementations if not available
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from olmocr import render_pdf_to_base64png, get_anchor_text, build_finetuning_prompt
    has_models = True
except ImportError:
    has_models = False
    print("Warning: transformers or olmocr not installed. Using mock implementations.")

# Initialize model + processor globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model with conditional device mapping only if models are available
if has_models:
    if torch.cuda.is_available():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float32,
            device_map="auto"
        ).eval()
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float32
        ).to(device).eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
else:
    model = None
    processor = None
    print("Models not loaded - using mock mode")

# Initialize Flask app
app = Flask(__name__)
if has_cors:
    CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'temp_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit as per requirements
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}  # Add image support

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_document_with_olmocr(filepath):
    print(f"Running OlmOCR model on file: {filepath}")

    # If models are not available, return mock data
    if not has_models:
        print("Using mock implementation since models are not available")
        return """This is a mock OCR output.

Document contains:
- Name: John Doe
- Date: 2023-05-15
- Address: 123 Main Street, Anytown, USA

Table of data:
| Item | Quantity | Price |
| Product A | 2 | $10.99 |
| Product B | 1 | $24.50 |
"""

    # Handle both PDFs and images
    if filepath.lower().endswith('.pdf'):
        image_base64 = render_pdf_to_base64png(filepath, 1, target_longest_image_dim=1024)
        anchor_text = get_anchor_text(filepath, 1, pdf_engine="pdfreport", target_length=4000)
    else:
        # For images, convert to base64
        with open(filepath, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        anchor_text = ""  # No anchor text for images

    prompt = build_finetuning_prompt(anchor_text)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    output = model.generate(
        **inputs,
        temperature=0.8,
        max_new_tokens=512,
        num_return_sequences=1,
        do_sample=True,
    )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return text_output[0]

def parse_olmocr_output(olmocr_raw_output):
    structured_data = {
        "entities": {
            "names": [],
            "dates": [],
            "addresses": []
        },
        "tables": [],
        "raw_text": olmocr_raw_output
    }

    # Extract names (multiple patterns)
    name_patterns = [
        r'(Customer Name|Patient Name|Name):\s*(.+)',
        r'([A-Z][a-z]+ [A-Z][a-z]+)',  # Simple name pattern
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, olmocr_raw_output, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                name = match[1].strip()
            else:
                name = match.strip()
            if name and name not in structured_data["entities"]["names"]:
                structured_data["entities"]["names"].append(name)

    # Extract dates (multiple formats)
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
        r'(\d{1,2}/\d{1,2}/\d{2,4})',  # M/D/YY or M/D/YYYY
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, olmocr_raw_output)
        for match in matches:
            if match not in structured_data["entities"]["dates"]:
                structured_data["entities"]["dates"].append(match)

    # Extract addresses
    address_patterns = [
        r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Way|Circle|Cir|Terrace|Ter))',
        r'Address:\s*(.+)',
    ]
    for pattern in address_patterns:
        matches = re.findall(pattern, olmocr_raw_output, re.IGNORECASE)
        for match in matches:
            if match and match not in structured_data["entities"]["addresses"]:
                structured_data["entities"]["addresses"].append(match.strip())

    # Table extraction (improved)
    lines = olmocr_raw_output.split('\n')
    table_data = []
    in_table = False
    headers = []
    
    for line in lines:
        line = line.strip()
        if '|' in line and any(char.isdigit() for char in line):
            if not in_table:
                # Extract headers
                headers = [h.strip() for h in line.split('|')]
                in_table = True
            else:
                # Extract row data
                row_data = [cell.strip() for cell in line.split('|')]
                if len(row_data) == len(headers):
                    table_data.append(row_data)
        elif in_table and not line:
            break
    
    if table_data and headers:
        structured_data["tables"].append({
            "headers": headers,
            "rows": table_data
        })

    return structured_data

@app.route('/')
def home():
    return "Backend is running! Send files to /extract."

@app.route('/extract', methods=['POST'])
def extract_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = None
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            print(f"Processing file: {filepath}")
            olmocr_output = process_document_with_olmocr(filepath)
            print("OlmOCR processing complete. Parsing output...")

            structured_data = parse_olmocr_output(olmocr_output)

            return jsonify({
                'message': 'File processed successfully!',
                'filename': filename,
                'extracted_data': structured_data
            }), 200
        else:
            return jsonify({'error': 'File type not allowed. Supported: PDF, PNG, JPG, JPEG'}), 400

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up temporary file: {filepath}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
