import cv2
import requests
from PIL import Image
from io import BytesIO
import base64
import json

# Video dosyasından istenilen kareyi alacak fonksiyon
def extract_frame_at_position(video_path, frame_pos):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if not ret:
        return None
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return image

# Video dosyasının FPS ve süresini hesaplayan fonksiyon
def get_video_duration_and_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration, fps

# VILA API'ye görsel ve prompt'u gönderen fonksiyon
def send_to_vila_api(image, prompt):
    # Görseli base64 formatına çevir
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()

    # API URL ve header bilgileri
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"
    headers = {
        "Authorization": "Bearer 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",
        "Accept": "application/json"
    }

    # API'ye gönderilecek payload
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

    # API çağrısını yapma
    response = requests.post(invoke_url, headers=headers, json=payload)

    # Yanıtı JSON formatında döndürme
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API hatası: {response.status_code}"}

# Ana işlem fonksiyonu
def process_video(video_path, frame_pos, prompt):
    # Videonun FPS ve süresini al
    duration, fps = get_video_duration_and_fps(video_path)
    print(f"Video Süresi: {duration}s, FPS: {fps}")

    # Belirtilen kareyi al
    image = extract_frame_at_position(video_path, frame_pos)
    if image:
        print(f"Video'dan {frame_pos} numaralı kare alındı.")

        # VILA API'ye gönder ve sonucu al
        result = send_to_vila_api(image, prompt)

        # JSON formatında sonucu yazdır
        print(json.dumps(result, indent=2))
    else:
        print("Kare alınamadı.")

# Kullanıcıdan video yolu, kare pozisyonu ve prompt al
if __name__ == "__main__":
    video_path = input("Videonun yolu: ")  # Video dosyasının yolu
    frame_pos = int(input("Hangi kareyi almak istiyorsunuz? (Başlangıç kareyi): "))  # Hedef kare pozisyonu
    prompt = input("Prompt'u girin: ")  # API'ye gönderilecek prompt

    process_video(video_path, frame_pos, prompt)
