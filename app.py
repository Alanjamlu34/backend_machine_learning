from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io
import base64
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
CORS(app)

# --- SETUP NLP ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- LOAD ASSETS ---
model_image = tf.keras.models.load_model('image_clasification.h5')
model_nlp = tf.keras.models.load_model('Sentiment_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# ==========================================
# 1. ENDPOINTS PREPROCESSING
# ==========================================

@app.route('/preprocess/image', methods=['POST'])
def preprocess_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # 1. Load dan otomatis resize (Crop & Resize ke 224x224)
    img_bytes = io.BytesIO(file.read())
    img = load_img(img_bytes, target_size=(224, 224))
    
    # 2. Proses untuk Return Gambar (Base64)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG") # Simpan objek image ke buffer
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 3. Proses untuk Return Array (Rescale 1/255)
    img_array = img_to_array(img)
    img_array = img_array / 255.0 
    
    return jsonify({
        'cropped_image_base64': img_str,           # Gambar untuk ditampilkan di frontend
        'preprocessed_array': img_array.tolist()    # Array untuk dikirim ke endpoint predict
    })

@app.route('/preprocess/nlp', methods=['POST'])
def preprocess_nlp():
    data = request.get_json() if request.is_json else request.form
    print('ini katanya\n',data)
    text = data.get('text', '')

    # Clean text logic
    text_lower = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text_lower)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    clean_text = ' '.join([word for word in text.split() if word not in stop_words])

    # Convert ke sequence & padding
    sequences = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    return jsonify({
        'text_lower': text_lower,
        'clean_text': clean_text,
        'padded_sequence': padded.tolist()
    })

# ==========================================
# 2. ENDPOINTS PREDICT
# ==========================================

@app.route('/predict/image', methods=['POST'])
def predict_image():
    data = request.get_json()
    # Menerima array hasil preprocessing
    img_array = np.array(data.get('preprocessed_array'))
    img_array = np.expand_dims(img_array, axis=0) # Tambah batch dimension
    
    prediction = model_image.predict(img_array)
    print("Prediction Image: ", prediction)
    label = "Anjing" if prediction[0][0] > 0.5 else "Kucing"
    
    return jsonify({'result': label, 'confidence': float(prediction[0][0])})

@app.route('/predict/nlp', methods=['POST'])
def predict_nlp():
    data = request.get_json()
    # Menerima padded_sequence hasil preprocessing
    padded = np.array(data.get('padded_sequence'))
    
    prediction = model_nlp.predict(padded)
    labels = ['Negative', 'Neutral', 'Positive']
    idx = np.argmax(prediction)
    
    return jsonify({
        'sentiment': labels[idx],
        'confidence': float(np.max(prediction))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
