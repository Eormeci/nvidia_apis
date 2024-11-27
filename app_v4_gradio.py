# gradio_app.py
import gradio as gr
import requests
import io

# Flask server URL'ini belirliyoruz
FLASK_URL = 'http://localhost:5000/predict'

# Gradio arayüzü fonksiyonu
def predict(image, prompt):
    try:
        # Resmi BytesIO kullanarak bytes formatına çeviriyoruz
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)  # BytesIO pointer'ını başa alıyoruz

        # POST isteği gönderiyoruz
        files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {'prompt': prompt}
        
        # Flask'a POST isteği gönder
        response = requests.post(FLASK_URL, files=files, data=data)
        
        # Yanıtı kontrol ediyoruz
        if response.status_code == 200:
            # Flask'tan dönen yanıtı Gradio arayüzünde gösteriyoruz
            return response.json()['response']
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
