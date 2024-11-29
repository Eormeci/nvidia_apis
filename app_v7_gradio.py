import gradio as gr
import requests
import io
import uuid
import base64
import json
from PIL import Image
import cv2
import numpy as np

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

# Videodan o anki kareyi alacak fonksiyon
def extract_frame_at_position(video_path, frame_pos):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if not ret:
        return None
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return image

# Videonun FPS'ini ve süresini hesaplayan fonksiyon
def get_video_duration_and_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    return duration, fps

def frame_to_base64(frame):
    try:
        # Convert frame to numpy array if it's not already
        frame_np = np.array(frame)
        
        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame_np)
        
        # Convert to base64, ensuring it's a bytes object
        base64_frame = base64.b64encode(buffer)
        
        return base64_frame  # This is now a bytes object
    except Exception as e:
        print(f"Error in frame_to_base64: {e}")
        raise

    
# Gradio arayüzü fonksiyonu
def predict(video, prompt):
    try:
        # Check if video is a file path or a byte-like object
        if isinstance(video, str):
            # If it's a string (file path), open and read the file
            with open(video, 'rb') as f:
                video_data = f.read()
        elif hasattr(video, 'read'):
            # If it's a file-like object, read its contents
            video_data = video.read()
        else:
            # If it's already bytes, use it directly
            video_data = video

        # Video'nun kaynağını geçici bir dosyaya kaydediyoruz
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_data)

        # Video'nun FPS'ini alıyoruz
        duration, fps = get_video_duration_and_fps(video_path)
        
        # Video'nun o anki karesini almak için frame pozisyonunu hesaplıyoruz
        current_frame_pos = int(fps * duration)  # Videonun son karesini alıyoruz
        frame_image = extract_frame_at_position(video_path, current_frame_pos)

        if frame_image is None:
            return "Error: Could not extract the frame."

        # Frame'i Base64 formatına çeviriyoruz
        base64_frame = frame_to_base64(frame_image)  # This will be bytes

        # POST isteği için gerekli verileri hazırlıyoruz
        data = {
            'model': 'Efficient-Large-Model/VILA1.5-3b',  # Model adı
            'messages': [
                {"role": "user", "content": prompt},
                {"role": "user", "image": base64_frame.decode('utf-8')}  # Now explicitly decoding bytes to string
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
        print(f"Detailed error while making request: {str(e)}")
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        return f"Error: {str(e)}"

# Gradio arayüzü oluşturuluyor
with gr.Blocks() as demo:
    gr.Markdown("### Video Tanımlama Modeli")
    
    with gr.Row():
        with gr.Column():
            # Kullanıcıdan video alıyoruz
            video_input = gr.Video(label="Video Yükle")  # Video dosyasını alıyoruz
            prompt_input = gr.Textbox(label="Prompt Girin", placeholder="Örneğin: Describe the current frame of the video.")
            submit_button = gr.Button("Gönder")
        
        with gr.Column():
            output_text = gr.Textbox(label="Model Yanıtı", interactive=False)

    # Submit butonuna tıklanınca yapılacak işlem
    def process_video_frame(video, prompt):
        # Video ve promptu predict fonksiyonuna gönder
        result = predict(video, prompt)
        return result

    submit_button.click(process_video_frame, inputs=[video_input, prompt_input], outputs=output_text)

# Arayüzü başlatıyoruz
demo.launch()
