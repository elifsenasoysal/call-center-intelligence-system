import pandas as pd
import json
import os
import random # Ses seviyelerine doğallık katmak için eklendi

def calculate_dynamic_volume(text):
    """
    Metnin içeriğine, noktalamasına ve uzunluğuna göre dinamik bir ses seviyesi (dB) belirler.
    Sıradan bir konuşma genelde -18 ile -14 dB arasındadır.
    """
    text_lower = text.lower()
    
    # Başlangıç için ortalama bir ses seviyesi seç (rastgelelik ile)
    base_volume = random.uniform(-17.5, -14.5)
    
    # 1. Yüksek Ses Belirtileri (Sinir, Şikayet, Vurgu)
    if "!" in text or text.isupper():
        base_volume += random.uniform(3.0, 6.0) # Sesi yükselt (0'a yaklaştır)
    if any(word in text_lower for word in ["şikayet", "iptal", "sorun", "hayır", "yavaş", "bozuk"]):
        base_volume += random.uniform(1.5, 3.5)
        
    # 2. Düşük Ses Belirtileri (Düşünme, Kararsızlık, Sakinlik)
    if "..." in text or any(word in text_lower for word in ["hmm", "şey", "bilmiyorum", "belki"]):
        base_volume -= random.uniform(2.0, 4.5) # Sesi kıs (- yönünde büyüt)
        
    # 3. Kısa Onaylamalar genelde daha sessiz olur ("evet", "tamam")
    if len(text) < 15 and any(word in text_lower for word in ["evet", "tamam", "peki", "olur"]):
        base_volume -= random.uniform(1.0, 2.5)

    # Çok patlayan veya çok kısılan sesleri sınırla (Gerçekçilik için)
    if base_volume > -5.0:
        base_volume = random.uniform(-7.0, -5.0)
    elif base_volume < -25.0:
        base_volume = random.uniform(-25.0, -23.0)
        
    return round(base_volume, 1)

def csv_to_audio_json(csv_path, output_path):
    print(f"📂 Veri seti okunuyor: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ HATA: {csv_path} bulunamadı!")
        return

    print("🔄 Dinamik ses seviyeli özel JSON formatına dönüştürülüyor...")
    
    all_calls = []
    
    for index, row in df.iterrows():
        turkish_text = str(row.get('turkish_text', ''))
        
        if pd.isna(row.get('turkish_text')) or not turkish_text.strip():
            continue
            
        lines = turkish_text.strip().split('\n')
        
        utterances = []
        current_time = 0.0 
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Temsilci:'):
                speaker = "agent"
                text = line.replace('Temsilci:', '').strip()
            elif line.startswith('Müşteri:'):
                speaker = "customer"
                text = line.replace('Müşteri:', '').strip()
            else:
                continue 
                
            duration = round(len(text) * 0.05, 1)
            
            # Ses seviyesini sabit vermek yerine yeni zeki fonksiyonumuzu çağırıyoruz
            dynamic_volume = calculate_dynamic_volume(text)
            
            utterance = {
                "speaker": speaker,
                "start_time": round(current_time, 1),
                "end_time": round(current_time + duration, 1),
                "text": text,
                "volume_db": dynamic_volume 
            }
            utterances.append(utterance)
            
            current_time += duration + random.uniform(0.3, 0.8) # Nefes alma süresini de biraz rastgele yaptık

        call_doc = {
            "file_name": f"call_{index + 1:03d}.wav",
            "language": "tr",
            "duration_seconds": round(current_time, 1),
            "utterances": utterances
        }
        
        all_calls.append(call_doc)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_calls, f, ensure_ascii=False, indent=2)
        
    print(f"✅ İşlem tamam! {len(all_calls)} adet çağrı şu formata dönüştürüldü: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    INPUT_CSV = os.path.join(base_dir, "data", "turkish_telecom_dataset.csv")
    OUTPUT_JSON = os.path.join(base_dir, "data", "train_data.json")
    
    csv_to_audio_json(INPUT_CSV, OUTPUT_JSON)