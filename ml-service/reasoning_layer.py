
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
reasoning_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")

def reason_about_scene(fused_data: dict, question: str = "What can be inferred from this audio?") -> str:
    transcript_summary = " | ".join(
        f'{seg["speaker"]} ({seg.get("language","unknown")}): "{seg["text"]}"'
        for seg in fused_data.get("transcript_segments", [])
    )
    bg_summary = ", ".join(fused_data.get("background_sounds", [])) or "none detected"
    prompt = f"""<|user|>
You are an audio scene understanding assistant. Given detected signals from an audio clip, reason about the situation like a human listening would.

Speech transcript: {transcript_summary or "no speech detected"}
Background sounds detected: {bg_summary}
Dominant emotion: {fused_data.get("emotion", "unknown")}
Background noise class: {fused_data.get("noise_class", "unknown")}

Question: {question}
Answer concisely in 2-3 sentences. Quote any numbers, flight codes, or names from the transcript exactly as given.<|end|>
<|assistant|>"""
    inputs = tokenizer(prompt, return_tensors="pt").to(reasoning_model.device)
    output = reasoning_model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
