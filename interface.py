import gradio as gr
import cv2
from PIL import Image
import requests
from io import BytesIO

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

# VILA API'ye görsel ve prompt'u gönderen fonksiyon
def send_to_vila_api(image, prompt):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    files = {'image': ('frame.jpg', image_bytes, 'image/jpeg')}
    data = {'prompt': prompt}
    response = requests.post("http://127.0.0.1:5003/analyze_vila", files=files, data=data)
    if response.status_code == 200:
        return response.json().get('result', 'Yanıt alınamadı.')
    else:
        return f"API hatası: {response.status_code}"

# NEVA API'sini çağıran Flask API'ye görsel ve prompt'u gönderen fonksiyon
def send_to_neva_api(image, prompt):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    files = {'image': ('frame.jpg', image_bytes, 'image/jpeg')}
    data = {'prompt': prompt}
    response = requests.post("http://127.0.0.1:5003/analyze_neva", files=files, data=data)
    if response.status_code == 200:
        return response.json().get('result', 'Yanıt alınamadı.')
    else:
        return f"API hatası: {response.status_code}"

# Gradio arayüzü
def gradio_interface():
    with gr.Blocks() as demo:
        # Model seçimi için dropdown
        model_choice = gr.Dropdown(choices=["VILA", "NEVA-22B"], label="Model Selection", value="VILA")

        # Video yükleme ve prompt girişi
        video_input = gr.Video(label="Videoyu Yükleyin")
        prompt_input = gr.Textbox(label="Görsele Dair Soru Prompt'u")

        # Çıkışlar için alanlar
        output_image = gr.Image(label="Seçilen Kare")
        output_text = gr.Textbox(label="Model Yanıtı")

        # Gönderme butonu
        submit_button = gr.Button("API'ya Gönder ve Yorumla")

        # Videodan anlık kareyi alıp API'ye gönderme
        def process_frame_from_video(video, prompt, model):
            # Videonun FPS ve süresini al
            duration, fps = get_video_duration_and_fps(video)
            frame_position = int(duration * fps)  # Son kareyi almak için
            image = extract_frame_at_position(video, frame_position)
            if image:
                if model == "VILA":
                    result = send_to_vila_api(image, prompt)
                elif model == "NEVA-22B":
                    result = send_to_neva_api(image, prompt)
                else:
                    result = "Bilinmeyen model seçimi."
                return result, image
            else:
                return "Anlık kare alınamadı.", None

        # Butona tıklandığında işlemi başlat
        submit_button.click(process_frame_from_video, inputs=[video_input, prompt_input, model_choice], outputs=[output_text, output_image])

    demo.launch()

# Arayüzü başlat
gradio_interface()
