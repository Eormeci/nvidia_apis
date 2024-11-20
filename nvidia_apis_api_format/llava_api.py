import subprocess
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import re
import logging

app = Flask(__name__)

# İstenen metin bloğunu ayıklamak için işlev
def extract_text_between(output):
    # encode_image_with_clip ile başlayıp llama_print_timings ile biten kısmı seçiyoruz
    match = re.search(
        r"encode_image_with_clip:.*?\n\n(.*?)\n\nllama_print_timings:", 
        output, 
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    else:
        return "İstenen içerik bulunamadı."

# Llava modelini çağıran fonksiyon
def run_llava_with_custom_inputs(image_path, prompt_text):
    # Komutu oluşturma
    command = (
        f"./jetson-containers run --workdir=/opt/llama.cpp/bin "
        f"--volume {image_path}:/data/images/cat.jpg "
        f"$(./autotag llama_cpp:gguf) "
        f"/bin/bash -c './llava-cli "
        f"--model $(huggingface-downloader mys/ggml_llava-v1.5-7b/ggml-model-q4_k.gguf) "
        f"--mmproj $(huggingface-downloader mys/ggml_llava-v1.5-7b/mmproj-model-f16.gguf) "
        f"--n-gpu-layers 999 "
        f"--image /data/images/cat.jpg "
        f"--prompt \"{prompt_text}\"'"
    )

    try:
        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            output, errors = process.communicate()
            if process.returncode == 0:
                # İlgili metin bloğunu ayıkla
                return extract_text_between(output)
            else:
                return f"Hata oluştu: {errors.strip()}"
    except Exception as e:
        return f"Bir hata oluştu: {e}"
    

logging.basicConfig(level=logging.DEBUG)
# Llava API endpoint'i
@app.route('/analyze_llava', methods=['POST'])
def analyze_llava():
    try:
        image_file = request.files['image']
        prompt = request.form['prompt']
        image = Image.open(image_file)
        
        # Görseli geçici olarak kaydetme
        temp_image_path = "/tmp/llava_temp_image.jpg"
        image.save(temp_image_path)
        
        # Llava'yı çağırma
        result = run_llava_with_custom_inputs(temp_image_path, prompt)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5003, debug=True)
