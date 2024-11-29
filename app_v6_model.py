import os
import io
import base64
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser
import time

# Flask uygulaması başlatılıyor
app = Flask(__name__)

# Flask ile model yükleme
def load_model():
    args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
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
    return model

# Modeli yüklüyoruz
model = load_model()

@app.route('/v1/chat/completions', methods=['POST'])
def predict():
    try:
        # OpenAI uyumlu API'de gelen parametreleri alıyoruz
        data = request.get_json()

        # Model adı ve mesajları alıyoruz
        model_name = data.get('model', "Efficient-Large-Model/VILA1.5-3b")
        messages = data.get('messages', [])

        # Mesajları kontrol et
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Mesajlardan son kullanıcının prompt'unu ve görseli alıyoruz
        image = None
        prompt = None
        for message in messages:
            if 'image' in message:
                # Base64 formatındaki görseli çözümlüyoruz
                image_data = message['image']
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))  # Base64'ten görsele dönüştürüyoruz
            elif 'content' in message:
                prompt = message['content']
        
        # Görseli işleyelim (eğer varsa)
        if image:
            img = np.array(image)

            # Modelin işlemi için gerekli olan chat_history oluşturuluyor
            chat_history = ChatHistory(model)
            chat_history.append('user', image=img)
        
        # Kullanıcıdan gelen prompt'a göre modelin cevabını almak için
        if prompt:
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

            # Yanıtı OpenAI formatında dönüyoruz
            return jsonify({
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': model_name,
                'choices': [{'message': {'role': 'assistant', 'content': response}, 'finish_reason': 'stop'}]
            })
        else:
            return jsonify({'error': 'No prompt found'}), 400
    
    except Exception as e:
        # Hata durumunda daha ayrıntılı bilgi veriyoruz
        return jsonify({'error': f"Error processing the request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
