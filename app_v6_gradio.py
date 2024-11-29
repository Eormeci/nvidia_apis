import gradio as gr
import requests
import io
import uuid
import base64
import json

# Flask server URL'ini belirliyoruz
FLASK_URL = 'http://localhost:5000/v1/chat/completions'  # API endpointini güncelledik

# JSON formatında çıktı üretme fonksiyonu
def create_openai_compatible_json(content):
    return {
        "id": str(uuid.uuid4()),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "completion_tokens": len(content.split()),
            "prompt_tokens": 0,
            "total_tokens": len(content.split())  # Basit bir token sayımı
        }
    }

# Görseli Base64 formatına çevirme fonksiyonu
def image_to_base64(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')  # Base64 string olarak döndürüyoruz

# Gradio arayüzü fonksiyonu
def predict(image, prompt):
    try:
        # Görseli Base64 formatına çeviriyoruz
        base64_image = image_to_base64(image)

        # POST isteği için gerekli verileri hazırlıyoruz
        data = {
            'model': 'Efficient-Large-Model/VILA1.5-3b',  # Model adı
            'messages': [
                {"role": "user", "content": prompt},
                {"role": "user", "image": base64_image}  # Base64 formatında görseli gönderiyoruz
            ]
        }

        # Flask'a POST isteği gönder
        response = requests.post(FLASK_URL, json=data)

        # Yanıtı kontrol ediyoruz
        if response.status_code == 200:
            # Flask'tan dönen yanıtı Gradio arayüzünde gösteriyoruz
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error response from server: {response.status_code} - {response.text}")
            return f"Error: {response.json()['error']}"
    except Exception as e:
        print(f"Error while making request: {str(e)}")
        return f"Error: {str(e)}"

# Gradio arayüzü oluşturuluyor
with gr.Blocks() as demo:
    gr.Markdown("### Görsel Tanımlama Modeli")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Resim Yükle")
            prompt_input = gr.Textbox(label="Prompt Girin", placeholder="Örneğin: Describe the image.")
            submit_button = gr.Button("Gönder")
        with gr.Column():
            output_text = gr.Textbox(label="Model Yanıtı", interactive=False)
    
    submit_button.click(fn=predict, inputs=[image_input, prompt_input], outputs=output_text)

# Arayüzü başlatıyoruz
demo.launch()
