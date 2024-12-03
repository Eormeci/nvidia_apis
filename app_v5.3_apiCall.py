import os
import requests
import json

FLASK_URL = 'http://localhost:5000/predict'  # Flask server URL
image_path = "/home/openzeka/Desktop/cat.jpg"  # Görselin yolu

# OpenAI API formatında veri göndermek için
data = {
    "model": "Efficient-Large-Model/VILA-2.7b",  # Kullanmak istediğiniz model
    "messages": [
        {
            "role": "user",  # Bu mesaj kullanıcıdan geliyor
            "content": "Is this a cat or dog?"  # Kullanıcı tarafından girilen prompt
        }
    ],
    "temperature": 0.7,
    "max_tokens": 150,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": ["\n"]
}

# Görseli okuma ve JSON verisi ile gönderme
with open(image_path, 'rb') as image_file:
    files = {'image': ('image.jpg', image_file, 'image/jpeg')}  # Burada 'image' parametresi doğru olmalı

    # JSON verisini form verisi olarak göndermeliyiz (data=...)
    response = requests.post(FLASK_URL, files=files, data={'data': json.dumps(data)})

    # Yanıtı kontrol ediyoruz
    if response.status_code == 200:
        # Flask sunucusundan dönen yanıtı alıyoruz
        json_response = response.json()
        # Yanıtı yazdırıyoruz
        print("Generated response:", json_response['choices'][0]['text'])
    else:
        print(f"Error response from server: {response.status_code} - {response.text}")
