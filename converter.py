import json
import os

# Yolları belirliyoruz (converter.py ana dizinde olduğu için current_dir ana dizindir)
current_dir = os.path.dirname(__file__)
input_file = os.path.join(current_dir, "data", "train_data.json")

# Çıktı dosyasını nokta atışı Sena'nın kodunun yanına gönderiyoruz
output_dir = os.path.join(current_dir, "services", "analysis", "training")
output_file = os.path.join(output_dir, "train_data.jsonl")

print("🔄 Veri, Sena'nın formatına (.jsonl) çevriliyor...")

try:
    # Eğer training klasörü yoksa hata vermesin diye klasörü oluşturuyoruz (güvenlik önlemi)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for call in data:
            transcript = ""
            for u in call["utterances"]:
                transcript += f"[{u['start_time']}s] {u['speaker']} ({u['volume_db']} dB): {u['text']}\n"
            
            # Sena'nın kodunun aradığı format:
            jsonl_line = {
                "instruction": "Sen profesyonel bir çağrı merkezi kalite analisti yapay zekasın. Aşağıdaki saniye ve ses desibeli (dB) verileriyle verilmiş çağrı dökümünü analiz et.",
                "input": transcript.strip(),
                "output": "Çağrı analizi tamamlandı. Müşteri ve temsilci arasındaki iletişim kaydedilen ses desibelleri üzerinden incelenmiştir."
            }
            f_out.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
    print(f"✅ İşlem başarılı! Dosya tam olarak şuraya kargolandı: {output_file}")
except Exception as e:
    print(f"❌ Bir hata oluştu: {e}")