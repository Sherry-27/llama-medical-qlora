Llama-3.2-1B-Instruct Medical QLoRA

This repository contains a parameter-efficient fine-tuned medical reasoning model based on Meta Llama 3.2-1B-Instruct, trained using LoRA and QLoRA (4-bit quantization) on an English medical reasoning dataset.

The project demonstrates how high-quality medical reasoning performance can be achieved while training less than 0.5% of total model parameters, making it suitable for low-resource environments and research prototyping.

Model Overview

Base Model: meta-llama/Llama-3.2-1B-Instruct

Fine-Tuning Method: QLoRA (4-bit)

Trainable Parameters: ~4.2M (0.35%)

Language: English

Domain: Medical reasoning and clinical QA (educational use)

Developer: Shaheer Khan

The model is trained to generate step-by-step reasoning followed by a final answer, making it suitable for medical education, reasoning demonstrations, and research.

## Model & Training Resources

- **Hugging Face Model**:  
  https://huggingface.co/Shaheerkhan/llama-3.2-1b-medical-qlora

- **Kaggle Training Notebook**:  
  https://www.kaggle.com/code/shaheerkhan27/llm-lora-fine-tuning

Intended Use
Supported Use

Medical question answering (educational / research)

Clinical reasoning explanation

Medical AI prototyping

Instruction-following medical dialogue

Out-of-Scope Use

Real-world clinical diagnosis or treatment decisions

Autonomous medical advice

High-risk healthcare applications without expert supervision

This model is not clinically validated and must not be used for real medical decisions.

Dataset

Dataset: medical-o1-reasoning-SFT (English subset)

Total Samples: ~10,000

Split:

Train: 8,000

Validation: 1,000

Test: 1,000

Average Input Length: 156 tokens

Average Output Length: 89 tokens

Data Format
{
  "instruction": "Analyze the following medical case",
  "input": "A 62-year-old diabetic patient presents with...",
  "output": "Based on the symptoms, the differential diagnosis includes..."
}

Training Summary

Hardware: Kaggle T4 / P100 GPU (16GB VRAM)

Training Time: ~3–4 hours

Epochs: 2–3

Optimizer: paged_adamw_8bit

Precision: FP16

Quantization: NF4 4-bit (bitsandbytes)

LoRA Configuration
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

QLoRA Quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

Performance
Metric	Score
Task Accuracy	87.0%
F1 Score	0.85
ROUGE-L	0.78
Training Loss	0.42
Validation Loss	0.51
Comparison
Model	Accuracy	Trainable Params	VRAM
Base Llama-3.2-1B	52.3%	0	—
Full Fine-Tuning	89.2%	1.2B	42GB
LoRA (r=8)	84.8%	2.1M	14GB
QLoRA (r=16)	87.0%	4.2M	12GB
Installation
git clone https://github.com/Sherry-27/llama-lora-finetuning.git
cd llama-lora-finetuning
pip install -r requirements.txt

Requirements
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
bitsandbytes>=0.41.0
datasets>=2.14.0
trl>=0.7.0
accelerate>=0.24.0
wandb>=0.16.0
gradio>=4.0.0

Training
python train.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_r 16

Inference
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "Sherry-27/llama-3.2-1b-medical-qlora"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Sherry-27/llama-3.2-1b-medical-qlora"
)

prompt = "A patient presents with chest pain and shortness of breath."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Project Structure
llama-lora-finetuning/
├── data/
├── src/
├── notebooks/
├── outputs/
├── app.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md

Limitations & Risks

Not clinically validated

Possible hallucinations or incorrect reasoning

English-only training

Must be used with expert oversight

License

This project follows the Llama 3.2 Community License, consistent with the base model.

Citation
@misc{khan2025llama,
  author = {Shaheer Khan},
  title = {Parameter-Efficient Fine-Tuning of Llama 3.2 for Medical Reasoning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sherry-27/llama-lora-finetuning}
}

Contact

Shaheer Khan
GitHub: https://github.com/Sherry-27

Hugging Face: https://huggingface.co/Sherry-27

LinkedIn: https://www.linkedin.com/in/shaheer-khan-689a44265
