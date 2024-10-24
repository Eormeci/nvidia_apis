import time
from flask import Flask, request, jsonify
import requests, base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# VILA API çağrısını yapan fonksiyon
def analyze_image_vila(image, prompt):
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"
    stream = False

    # Görseli base64 formatına çevir
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()

    headers = {
        "Authorization": "Bearer your_api",
        "Accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 50,
        "stream": False,
    }

    with requests.Session() as session:  # Ayrı bir session açıyoruz
        response = session.post(invoke_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', 'Yanıt alınamadı.')
    else:
        return f"API hatası: {response.status_code}"

# NEVA API çağrısını yapan fonksiyon
def analyze_image_neva(image, prompt):
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"

    # Görseli base64 formatına çevir
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()

    headers = {
        "Authorization": "Bearer your_api",
        "Accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 50,
    }

    time.sleep(1)  # VILA API çağrısından sonra 1 saniye bekliyoruz
    
    with requests.Session() as session:  # Ayrı bir session açıyoruz
        response = session.post(invoke_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', 'Yanıt alınamadı.')
    else:
        return f"API hatası: {response.status_code}"

# Flask route'lar
@app.route('/analyze_vila', methods=['POST'])
def analyze_vila():
    try:
        image_file = request.files['image']
        prompt = request.form['prompt']
        image = Image.open(image_file)
        result = analyze_image_vila(image, prompt)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analyze_neva', methods=['POST'])
def analyze_neva():
    try:
        image_file = request.files['image']
        prompt = request.form['prompt']
        image = Image.open(image_file)
        result = analyze_image_neva(image, prompt)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
