import os
import torch
import gc
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Bu script'in bulunduğu dizin (training/)
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. ÇEVRE DEĞİŞKENLERİNİ YÜKLE
load_dotenv()

# 2. AYARLAR VE HAFIZA TEMİZLİĞİ
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise EnvironmentError(
        "HUGGINGFACE_TOKEN bulunamadı! "
        "Lütfen .env dosyasına HUGGINGFACE_TOKEN=hf_xxx şeklinde ekleyin."
    )
os.environ["HF_TOKEN"] = hf_token
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
        # EOS token: modelin cümlenin nerede biteceğini öğrenmesi için şart
        text = (
            f"### Talimat:\n{instruction}\n\n"
            f"### Giriş:\n{input}\n\n"
            f"### Yanıt:\n{output}"
        ) + tokenizer.eos_token
        texts.append(text)
    return texts

# 5. VERİ SETİNİ YÜKLE
# Absolute path kullanıyoruz — script nerede çalıştırılırsa çalıştırılsın doğru bulur
DATA_PATH = os.path.join(TRAINING_DIR, "train_data.jsonl")
try:
    dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
except FileNotFoundError:
    print(f"HATA: '{DATA_PATH}' dosyası bulunamadı!")
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
        output_dir = os.path.join(TRAINING_DIR, "outputs"),
    ),
)

# 7. EĞİTİMİ BAŞLAT
print("Eğitim başlıyor...")
trainer.train()

# 8. MODELİ KAYDET
# Absolute path: training/ klasörünün yanına kaydeder
MODEL_SAVE_PATH = os.path.join(TRAINING_DIR, "llama3_callcenter_model")
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"İşlem başarıyla tamamlandı! Model şuraya kaydedildi: {MODEL_SAVE_PATH}")