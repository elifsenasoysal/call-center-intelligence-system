import os
import torch
import gc
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import unsloth
from unsloth import FastLanguageModel

# 1. ÇEVRE DEĞİŞKENLERİNİ YÜKLE
load_dotenv()

# 2. AYARLAR VE HAFIZA TEMİZLİĞİ
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

gc.collect()
torch.cuda.empty_cache()

# 3. MODEL VE TOKENIZER YÜKLEME
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    device_map = "cuda",
)

# LoRA Adaptörlerini ekle
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. VERİ FORMATLAMA FONKSİYONU
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"### Talimat:\n{instruction}\n\n### Giriş:\n{input}\n\n### Yanıt:\n{output}"
        texts.append(text)
    return texts

# 5. VERİ SETİNİ YÜKLE 
try:
    dataset = load_dataset("json", data_files={"train": "train_data.jsonl"}, split="train")
except FileNotFoundError:
    print("HATA: 'train_data.jsonl' dosyası bulunamadı! Lütfen dosyanın script ile aynı dizinde olduğundan emin olun.")
    exit()

# 6. TRAINER KURULUMU
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_seq_length = max_seq_length,
    formatting_func = formatting_prompts_func,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 7. EĞİTİMİ BAŞLAT
print("Eğitim başlıyor...")
trainer.train()

# 8. MODELİ KAYDET
model.save_pretrained("llama3_callcenter_model")
tokenizer.save_pretrained("llama3_callcenter_model")
print("İşlem başarıyla tamamlandı! Model 'llama3_callcenter_model' klasörüne kaydedildi.")