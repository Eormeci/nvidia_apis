import gradio as gr
import cv2
from PIL import Image
import requests
from io import BytesIO


# Videodan belirli bir kareyi alacak fonksiyon
def extract_frame(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return None
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return image

# Videonun süresini hesaplayan fonksiyon
def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    return duration

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
    
    # Flask API'ye istek gönderiyoruz (NEVA API'yi çağıran API)
    response = requests.post("http://127.0.0.1:5003/analyze_neva", files=files, data=data)
    
    if response.status_code == 200:
        return response.json().get('result', 'Yanıt alınamadı.')
    else:
        return f"API hatası: {response.status_code}"

# Gradio arayüzü
def gradio_interface():
    with gr.Blocks() as demo:
        # Model seçimi için dropdown
        model_choice = gr.Dropdown(choices=["VILA", "NEVA-22B"], label="Model Seçimi", value="VILA")
        
        # Video yükleme ve zaman seçme
        video_input = gr.Video(label="Videoyu Yükleyin")
        time_input = gr.Slider(minimum=0, maximum=10, step=0.1, label="Zaman (saniye)", interactive=True)
        prompt_input = gr.Textbox(label="Görsele Dair Soru Prompt'u")

        # Çıkışlar için alanlar
        output_image = gr.Image(label="Seçilen Kare")
        output_text = gr.Textbox(label="Model Yanıtı")

        submit_button = gr.Button("Gönder")

        # Video ve zamanı işleyip kareyi göster ve seçilen modelin API'sine gönder
        def process_video_and_send(video, time, prompt, model):
            image = extract_frame(video, time)
            if image:
                if model == "VILA":
                    result = send_to_vila_api(image, prompt)
                elif model == "NEVA-22B":
                    result = send_to_neva_api(image, prompt)
                else:
                    result = "Bilinmeyen model seçimi."
                return result, image
            else:
                return "Belirtilen zamanda kare alınamadı.", None

        # Video yüklendiğinde kaydırıcının maksimum değerini videonun uzunluğuna göre ayarla
        def update_slider(video):
            duration = get_video_duration(video)
            return gr.update(maximum=duration)

        # Video yüklendiğinde zaman kaydırıcısını güncelle
        video_input.change(fn=update_slider, inputs=video_input, outputs=[time_input])

        # Butona tıklandığında işlem yapılacak ve kare gösterilecek
        submit_button.click(process_video_and_send, inputs=[video_input, time_input, prompt_input, model_choice], outputs=[output_text, output_image])

    demo.launch()

# Arayüzü başlat
gradio_interface()
