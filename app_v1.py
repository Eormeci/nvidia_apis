from nano_llm import NanoLLM

model = NanoLLM.from_pretrained(
   "meta-llama/Meta-Llama-3-8B",  # HuggingFace repo/model name, or path to HF model checkpoint
   api='mlc',                   # supported APIs are: mlc, awq, hf
   api_token='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',    # HuggingFace API key for authenticated models ($HUGGINGFACE_TOKEN)
   quantization='q4f16_ft'      # q4f16_ft, q4f16_1, q8f16_0 for MLC, or path to AWQ weights
)

response = model.generate("Once upon a time,", max_new_tokens=128)

for token in response:
   print(token)