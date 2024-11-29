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
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
        
        # Video'nun mevcut kare sayısını alalım
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_pos >= total_frames:
            print(f"Requested frame position {frame_pos} exceeds total frames {total_frames}.")
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to extract frame at position: {frame_pos}")
            return None
        
        # Çıkarılan kareyi RGB formatına çevirip PIL image olarak döndürüyoruz
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return image
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None

# Videonun FPS'ini ve süresini hesaplayan fonksiyon
def get_video_duration_and_fps(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return 0, 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps == 0:
            print(f"FPS is zero, can't process video.")
            return 0, 0
        
        duration = frame_count / fps if fps > 0 else 0
        return duration, fps
    except Exception as e:
        print(f"Error getting video info: {e}")
        return 0, 0

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
        video_path = "temp_video2.mp4"
        with open(video_path, "wb") as f:
            f.write(video_data)

        # Video'nun FPS'ini alıyoruz
        duration, fps = get_video_duration_and_fps(video_path)
        
        if duration == 0 or fps == 0:
            print(f"Invalid FPS or duration: {fps}, {duration}")
            return "Error: Unable to retrieve valid video duration or FPS.", None
        
        # Video'nun o anki karesini almak için frame pozisyonunu hesaplıyoruz
        current_frame_pos = int(fps * duration)-1  # Videonun son karesini alıyoruz
        frame_image = extract_frame_at_position(video_path, current_frame_pos)

        if frame_image is None:
            return "Error: Could not extract the frame.", None

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
            return response.json()['choices'][0]['message']['content'], frame_image
        else:
            print(f"Error response from server: {response.status_code} - {response.text}")
            return f"Error: {response.json()['error']}", None
    except Exception as e:
        print(f"Detailed error while making request: {str(e)}")
        import traceback
        traceback.print_exc()  # Bu, hatayı daha ayrıntılı şekilde gösterir
        return f"Error: {str(e)}", None

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
            image_output = gr.Image(label="Video Karesi", interactive=False)

    # Submit butonuna tıklanınca yapılacak işlem
    def process_video_frame(video, prompt):
        # Video ve promptu predict fonksiyonuna gönder
        result, frame = predict(video, prompt)
        return result, frame

    submit_button.click(process_video_frame, inputs=[video_input, prompt_input], outputs=[output_text, image_output])

# Arayüzü başlatıyoruz
demo.launch()
