import requests
import base64
import json

# Flask API URL'si (bu, modelinize localhost üzerinden bağlanır)
FLASK_URL = 'http://localhost:5000/v1/chat/completions'

def create_openai_compatible_json(prompt, image_path=None):
    """OpenAI uyumlu JSON formatında veri oluşturur."""
    
    # Eğer bir görsel yolu verilmişse, görseli Base64 formatına çeviriyoruz
    image_data = None
    if image_path:
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')

    # OpenAI uyumlu JSON formatı
    return {
        "model": "Efficient-Large-Model/VILA1.5-3b",  # Model adı
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "user",
                "image": image_data  # Base64 formatında görsel
            }
        ]
    }

def call_flask_api(prompt, image_path=None):
    """Flask API'ye istek gönderir ve yanıtı döndürür."""
    # OpenAI uyumlu JSON verisini oluşturuyoruz
    request_data = create_openai_compatible_json(prompt, image_path)

    try:
        # POST isteği gönderiyoruz
        response = requests.post(FLASK_URL, json=request_data)

        # Yanıtı kontrol ediyoruz
        if response.status_code == 200:
            return response.json()  # JSON formatında yanıt döndür
        else:
            print(f"Error response from server: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

def process_response(response_json):
    """API yanıtını işleyip, modelin cevabını döndürür."""
    if response_json:
        try:
            # Yanıttan modelin cevabını alıyoruz
            return response_json['choices'][0]['message']['content']
        except KeyError:
            print("Response format is unexpected or missing keys.")
            return None
    return "No valid response."

# Kullanıcıdan prompt alalım
prompt = "Describe the image below."
image_path = "Mount_Docker/cat.jpg"  # Görselin yolu (gerekirse boş bırakabilirsiniz)

# API çağrısını yapalım
response_json = call_flask_api(prompt, image_path)

# Yanıtı işleyelim
response_content = process_response(response_json)

# Yanıtı ekrana yazdıralım
if response_content:
    print(f"Model response: {response_content}")
else:
    print("No response from the model.")
