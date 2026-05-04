import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify, render_template
from src.predict import load_model, predict_ir
import tempfile

app = Flask(__name__)
model = load_model("models/best_model.keras")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        file.save(tmp.name)
        result = predict_ir(tmp.name, model)
    
    ir = result['probability'] * 30
    status = 'normal' if 5 <= ir <= 30 else 'abnormal'
    
    return jsonify({
        'probability': result['probability'],
        'label': result['label'],
        'ir_estimate': round(ir, 1),
        'status': status
    })

if __name__ == '__main__':
    app.run(debug=True)