import time
import torch
from transformers import pipeline

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct"
]

def run_test(model_id):
    print(f"\n--- Testing {model_id} ---")
    device = 0 if torch.cuda.is_available() else -1
    
    start_load = time.perf_counter()
    pipe = pipeline("text-generation", model=model_id, device=device, torch_dtype=torch.float16)
    print(f"Load time: {time.perf_counter() - start_load:.2f}s")
    
    messages = [
        {"role": "system", "content": "You are a book assistant. Provide a short 2-sentence summary."},
        {"role": "user", "content": "Tell me about 'The Great Gatsby' by F. Scott Fitzgerald."}
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Warmup
    pipe(prompt, max_new_tokens=20, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id)
    
    # Bench
    start_gen = time.perf_counter()
    outputs = pipe(prompt, max_new_tokens=100, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id)
    end_gen = time.perf_counter()
    
    response = outputs[0]['generated_text'][len(prompt):].strip()
    tokens = len(pipe.tokenizer.encode(response))
    
    print(f"Gen time: {end_gen - start_gen:.4f}s")
    print(f"Tokens: {tokens}")
    print(f"Speed: {tokens / (end_gen - start_gen):.2f} tokens/sec")
    print(f"Response: {response}")
    
    # Cleanup to save VRAM
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for mid in MODELS:
        run_test(mid)
