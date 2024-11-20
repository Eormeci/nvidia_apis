import subprocess

def run_llava_with_custom_inputs(image_path, prompt_text):
    # Komutu oluşturma
    command = (
        f"./jetson-containers run --workdir=/opt/llama.cpp/bin "
        f"--volume {image_path}:/data/images/cat.jpg "  # Resmi doğru şekilde bağlamak için
        f"$(./autotag llama_cpp:gguf) "
        f"/bin/bash -c './llava-cli "
        f"--model $(huggingface-downloader mys/ggml_llava-v1.5-13b/ggml-model-q4_k.gguf) "
        f"--mmproj $(huggingface-downloader mys/ggml_llava-v1.5-13b/mmproj-model-f16.gguf) "
        f"--n-gpu-layers 999 "
        f"--image /data/images/cat.jpg "  # Konteyner içindeki yol
        f"--prompt \"{prompt_text}\"'"
    )

    try:
        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            output, errors = process.communicate()  # Tam komut bitiminde tüm çıktıyı al
            print("Çıktı:")
            print(output.strip())  # Komutun standart çıktısını yazdır

            if errors:
                print("Hatalar:")
                print(errors.strip())  # Eğer hata varsa hata mesajını yazdır

            if process.returncode != 0:
                print("Komut çalıştırılırken bir hata oluştu.")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# Kullanıcı girdisi olarak resim yolu ve prompt metni
image_path = "/home/openzeka/nvidia_apis/cat.jpg"
prompt_text = "What do you see in the photo?"

# Komutu çalıştır
run_llava_with_custom_inputs(image_path, prompt_text)
