---
language: en
license: llama3.2
tags:
  - medical
  - llama-3.2
  - qlora
  - instruction-tuning
  - chain-of-thought
  - reasoning
base_model: meta-llama/Llama-3.2-1B-Instruct
pipeline_tag: text-generation
---

# Llama-3.2-1B-Instruct Medical QLoRA

Fine-tuned version of **Llama 3.2 1B Instruct** using **QLoRA** (4-bit quantization) on the English subset of the **medical-o1-reasoning-SFT** dataset.  
The model is trained to produce step-by-step chain-of-thought reasoning (wrapped in `<think>...</think>` tags) followed by final answers, making it useful for medical question answering, clinical reasoning support, and educational purposes.

## Model Details

- **Developed by**: Shaheer Khan
- **Shared by**: Shaheer Khan[](https://huggingface.co/Sherry-27)
- **Model type**: LoRA adapter for causal language model
- **Language(s)**: English (trained on English medical reasoning data)
- **License**: Llama 3.2 Community License (same as base model)
- **Finetuned from model**: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Training hardware**: Free Kaggle T4 / P100 GPU (16 GB VRAM)
- **Training time**: ~4–6 hours (subset of ~10,000 samples, 2 epochs)

## Intended Use

### Direct Use
- Medical question answering and reasoning
- Step-by-step clinical decision support (educational / research use)
- Prototyping medical dialogue or tutoring systems

### Out-of-Scope Use
- Clinical diagnosis, treatment recommendations, or medical advice without human oversight
- High-stakes healthcare decisions
- Use violating the Llama 3.2 Acceptable Use Policy

### Bias, Risks, and Limitations
- May inherit factual inaccuracies, hallucinations, or biases from base model and synthetic dataset.
- Not clinically validated or approved for medical use (no HIPAA/FDA compliance).
- English-only performance — limited capability on other languages.
- Outputs should **always** be verified by qualified medical professionals.

## How to Get Started

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "Sherry-27/llama-3.2-1b-medical-qlora")
tokenizer = AutoTokenizer.from_pretrained("Sherry-27/llama-3.2-1b-medical-qlora")

# Inference example
messages = [
    {"role": "user", "content": "A 55-year-old man presents with chest pain and shortness of breath. What is the most likely diagnosis and initial management?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
