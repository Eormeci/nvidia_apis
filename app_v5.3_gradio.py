import gradio as gr
import requests
import io
import json

# Flask server URL'ini belirliyoruz
FLASK_URL = 'http://localhost:5000/predict'

# Gradio arayüzü fonksiyonu
def predict(image, prompt):
    try:
        # Resmi BytesIO kullanarak bytes formatına çeviriyoruz
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)  # BytesIO pointer'ını başa alıyoruz

        # OpenAI API formatında veriyi hazırlıyoruz
        openai_payload = {
            "model": "Efficient-Large-Model/VILA-2.7b",  # Kullanılan model adı
            "prompt": prompt,  # Kullanıcıdan gelen prompt
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["\n"],
        }
        
        # Flask'a POST isteği göndermek için kullanılan parametreler
        files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
        # Burada 'data' kısmını JSON formatında gönderiyoruz
        data = {
            'data': json.dumps(openai_payload)  # JSON formatında payload
        }
        
        # Flask'a POST isteği gönderiyoruz
        response = requests.post(FLASK_URL, files=files, data=data)
        
        # Yanıtı kontrol ediyoruz
        if response.status_code == 200:
            # Flask'tan dönen yanıtı OpenAI API formatında alıyoruz
            json_response = response.json()

            # OpenAI API formatındaki yanıtı alıyoruz
            if 'choices' in json_response:
                # Yanıt text'ini choices içinden alıyoruz
                model_response = json_response['choices'][0]['text']
                
                # Model yanıtını Gradio arayüzünde göstereceğiz
                return model_response
            else:
                return f"Error: Invalid response format from server."
        else:
            print(f"Error response from server: {response.status_code} - {response.text}")
            return f"Error: {response.json().get('error', 'Unknown error')}"
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
