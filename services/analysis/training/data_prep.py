from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# 1. Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. Llama 3 Çeviri Fonksiyonu
def translate_text(text):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sen profesyonel bir çevirmensin. Aşağıdaki diyaloğun TAMAMINI, hiçbir cümleyi atlamadan Türkçeye çevir. 
Müşteri ve Temsilci arasındaki tüm konuşma akışını koru.
Sadece çeviriyi ver.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 2048, 
            temperature = 0.2,
            eos_token_id = tokenizer.eos_token_id
        )
    
    decoded = tokenizer.batch_decode(outputs)[0]
    translation = decoded.split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
    
    prefixes = ["İşte çeviri:", "Here is the translation:", "Çeviri:"]
    for prefix in prefixes:
        if translation.startswith(prefix):
            translation = translation.replace(prefix, "").strip()
            
    return translation

def process_row(row):
    try:
        data = row['original']
        messages = data['messages']
        
        formatted_chat = ""
        for msg in messages:
            role_label = "agent" if msg['role'] == 'agent' else "customer"
            speaker_prefix = "Müşteri" if role_label == "customer" else "Temsilci"
            formatted_chat += f"{speaker_prefix}: {msg['content']}\n"
        
        turkish_text = translate_text(formatted_chat)
        return turkish_text
    except Exception as e:
        print(f"Hata: {e}")
        return None

# 3. Fetch the Dataset
dataset = load_dataset("Ming-secludy/telecom-customer-support-synthetic-replicas")
df = pd.DataFrame(dataset['train'])

# 4. Start the Translation Process 
tqdm.pandas()
df_subset = df.copy() 
print("Çeviri başlıyor, lütfen bekleyin...")
df_subset['turkish_text'] = df_subset.progress_apply(process_row, axis=1)

# 5. Save
df_subset.to_csv("turkish_telecom_dataset.csv", index=False)
print("\nÇeviri başarıyla tamamlandı ve 'turkish_telecom_dataset.csv' olarak kaydedildi!")