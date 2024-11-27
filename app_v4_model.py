# flask_server.py
import os
import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts
import time

# Flask uygulaması başlatılıyor
app = Flask(__name__)

# Flask ile model yükleme
def load_model():
    args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
    prompts = load_prompts(args.prompt)
    if not prompts:
        prompts = ["Describe the image."]
    if not args.model:
        args.model = "Efficient-Large-Model/VILA1.5-3b"
    
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
    return model, prompts

# Modeli yüklüyoruz
model, prompts = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Resmi alıyoruz
        if 'image' not in request.files:
            return jsonify({'error': 'No image file part'}), 400
        img_file = request.files['image']
        
        # Resmi okuma ve kontrol
        img = Image.open(img_file.stream)
        img = np.array(img)

        # Modelin işlemi için gerekli olan chat_history oluşturuluyor
        chat_history = ChatHistory(model)

        # Kullanıcının prompt'larına göre modelden yanıt alıyoruz
        chat_history.append('user', image=img)
        
        # Yanıtların bulunduğu liste
        results = []
        
        for prompt in prompts:
            print(f"Processing prompt: {prompt}")
            chat_history.append('user', prompt, use_cache=True)
            embedding, _ = chat_history.embed_chat()
            
            # Modeli çalıştırıyoruz
            print(f"Generating response for: {prompt}")
            reply = model.generate(
                embedding,
                kv_cache=chat_history.kv_cache,
                max_new_tokens=150,
                temperature=0.7
            )
            response = ''.join(reply)
            print(f"Generated response: {response}")
            results.append(response)
        
        # Model çıktısını JSON formatında döndürme
        return jsonify({'response': ' '.join(results)})
    
    except Exception as e:
        # Hata durumunda daha ayrıntılı bilgi veriyoruz
        return jsonify({'error': f"Error processing the request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
