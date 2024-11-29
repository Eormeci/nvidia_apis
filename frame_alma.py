import gradio as gr
import cv2
import numpy as np
import time

# Zamanı başlatan global değişken
start_time = None
cap = None  # Video kaynağını global değişken olarak tutalım

# Video oynatıldığında çağrılacak fonksiyon
def on_video_play(video):
    global start_time, cap
    start_time = time.time()  # Video oynatılmaya başlandığında zamanı başlat

    # Video dosyasını aç (Burada video yolu, doğrudan string olarak gelir)
    cap = cv2.VideoCapture(video)  # video parametresi artık dosya yoludur

    if not cap.isOpened():
        return "Video açılamadı. Lütfen geçerli bir video yükleyin."

    return "Video oynatılmaya başlandı."

# Geçen süreyi hesaplayan fonksiyon ve o saniyedeki kareyi gösteren fonksiyon
def get_time_elapsed_and_frame():
    if start_time is not None and cap is not None:
        # Geçen zamanı hesapla
        elapsed_time = time.time() - start_time
        elapsed_time = round(elapsed_time, 2)
        
        # O anki kareyi almak için video dosyasının o saniyesine git
        frame_number = int(elapsed_time * cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            # Kareyi uygun formatta döndür
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = np.array(frame)
            return f"Geçen Süre: {elapsed_time:.2f} saniye", frame_image
        else:
            return "Kare alınamadı.", None
    else:
        return "Henüz video oynatılmadı.", None

# Gradio arayüzü
with gr.Blocks() as demo:
    with gr.Row():
        video_input = gr.Video(label="Video Yükle", interactive=True)  # Video yükleme
    with gr.Row():
        process_button = gr.Button("Kareyi Göster")  # Buton
        elapsed_time_output = gr.Textbox(label="Geçen Süre")  # Geçen süreyi gösterecek textbox
        frame_output = gr.Image(label="Anlık Kare")  # Anlık kareyi gösterecek

    # Video oynatıldığında zaman başlasın
    video_input.play(on_video_play, inputs=[video_input], outputs=[elapsed_time_output])

    # Butona tıklanınca geçen süreyi ve o saniyedeki kareyi göster
    process_button.click(
        get_time_elapsed_and_frame, 
        inputs=[], 
        outputs=[elapsed_time_output, frame_output]
    )

# Arayüzü çalıştır
demo.launch()
