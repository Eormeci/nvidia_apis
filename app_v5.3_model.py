import os
import io
import uuid
import time
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts
import json

# Flask uygulaması başlatılıyor
app = Flask(__name__)

# Flask ile model yükleme
def load_model():
    print("Loading model...")
    args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
    if not args.model:
        args.model = "Efficient-Large-Model/VILA-2.7b"
        args.quantization = "q4f16_1"
    
    # Model yükleniyor
    print(f"Loading model: {args.model}")
    model = NanoLLM.from_pretrained(
        args.model, 
        api=args.api,
        quantization=args.quantization, 
        max_context_len=args.max_context_len,
        vision_model=args.vision_model,
        vision_scaling=args.vision_scaling, 
    )
    return model

# Modeli yüklüyoruz
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gelen veriyi kontrol edelim
        print("Incoming request...")
        
        # Resmi alıyoruz
        if 'image' not in request.files:
            print("Error: No image file part.")
            return jsonify({'error': 'No image file part'}), 400
        img_file = request.files['image']
        print("Image received.")
        
        # JSON formatında prompt alıyoruz
        data = request.form.get('data')  # raw JSON string from the form
        if not data:
            print("Error: No JSON data received.")
            return jsonify({'error': 'No JSON data received'}), 400
        
        # JSON verisini çözümle
        data = json.loads(data)  # JSON string'i çözümleme
        
        # Burada sadece 'prompt' anahtarını alıyoruz
        prompt = data.get("prompt", "Is this a cat or dog?")  # Varsayılan olarak bir prompt belirliyoruz
        print(f"Prompt received: {prompt}")

        # Resmi okuma ve kontrol
        img = Image.open(img_file.stream)
        img = np.array(img)

        # Modelin işlemi için gerekli olan chat_history oluşturuluyor
        chat_history = ChatHistory(model)

        # Kullanıcının prompt'larına göre modelden yanıt alıyoruz
        chat_history.append('user', image=img)
        
        # Yanıtların bulunduğu liste
        results = []
        
        print(f"Processing prompt: {prompt}")
        chat_history.append('user', prompt, use_cache=True)
        embedding, _ = chat_history.embed_chat()
        
        # Modeli çalıştırıyoruz
        print(f"Generating response for: {prompt}")
        reply = model.generate(
            embedding,
            streaming=True, 
            kv_cache=chat_history.kv_cache,
            max_new_tokens=150,
            temperature=0.7
        )
        response = ''.join(reply)
        print(f"Generated response: {response}")
        results.append(response)
        
        # OpenAI API formatına uygun response yapısı
        openai_response = {
            "id": str(uuid.uuid4()),  # Unique ID
            "object": "text_completion",
            "created": int(time.time()),  # Timestamp
            "model": "Efficient-Large-Model/VILA-2.7b",  # Model ismi
            "choices": [
                {
                    "text": ' '.join(results),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"  # Bu parametreyi uygun şekilde güncelleyebilirsiniz.
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Bu değeri hesaplayarak doğru şekilde ekleyebilirsiniz
                "completion_tokens": len(response.split()),  # Cevap token sayısı
                "total_tokens": len(response.split())  # Toplam token sayısı
            }
        }
        
        # Model çıktısını JSON formatında döndürüyoruz
        return jsonify(openai_response)
    
    except Exception as e:
        # Hata durumunda daha ayrıntılı bilgi veriyoruz
        print(f"Error processing the request: {str(e)}")
        return jsonify({'error': f"Error processing the request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
