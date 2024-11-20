import cv2
import requests
from PIL import Image
from io import BytesIO
import base64
import json
import time

def extract_frame_at_position(video_path, frame_pos):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if not ret:
        return None
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return image

def get_video_duration_and_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration, fps

def send_to_vila_api(image, prompt):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()

    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"
    headers = {
        "Authorization": "Bearer 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",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Connection": "keep-alive"
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
        "stream": True,
        "model": "vila-2-1"
    }

    try:
        session = requests.Session()
        response = session.post(invoke_url, headers=headers, json=payload, stream=True)
        
        if response.status_code != 200:
            return None
            
        return response
    except requests.exceptions.RequestException as e:
        return None

def process_streaming_response(response):
    if not response or not response.ok:
        return None

    full_response = ""
    start_time = time.time()
    
    try:
        buffer = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                buffer += chunk.decode('utf-8')
                lines = buffer.split('\n')
                buffer = lines[-1]
                
                for line in lines[:-1]:
                    if line.strip():
                        try:
                            if line.startswith('data: '):
                                json_str = line[6:].strip()
                                
                                if json_str == '[DONE]':
                                    break
                                
                                data = json.loads(json_str)
                                
                                if 'choices' in data and len(data['choices']) > 0:
                                    choice = data['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        content = choice['delta']['content']
                                        full_response += content
                                        
                                        if choice.get('finish_reason'):
                                            break
                            
                        except json.JSONDecodeError as e:
                            continue
                        except Exception as e:
                            continue

        return full_response

    except Exception as e:
        return None

def process_video(video_path, frame_pos, prompt):
    duration, fps = get_video_duration_and_fps(video_path)

    image = extract_frame_at_position(video_path, frame_pos)
    if image:
        response = send_to_vila_api(image, prompt)
        if response:
            full_text = process_streaming_response(response)
            if full_text:
                # Kelimeleri tek tek ve aralarına sleep ekleyerek yazdırma
                for word in full_text.split():
                    print(word, end=' ', flush=True)
                    time.sleep(0.05)  # Her kelime arasında 0.1 saniye bekle
                print()  # Yeni satıra geç
            else:
                print("[HATA] Metin işlenemedi")
        else:
            print("[HATA] API yanıtı alınamadı")
    else:
        print("[HATA] Kare alınamadı")

if __name__ == "__main__":
    try:
        # Sabit girişler
        video_path = "ornek_video.mp4"
        frame_pos = 4
        prompt = "describe"

        process_video(video_path, frame_pos, prompt)
    except Exception as e:
        print(f"[HATA] Program çalışırken hata oluştu: {e}")
