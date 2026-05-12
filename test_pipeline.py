import json
from services.analysis.pipeline import get_analysis

# Örnek STT çıktısı (shared/contracts.py formatına uygun)
sample_stt = {
    "file_name": "test_call.wav",
    "language": "tr",
    "duration_seconds": 45.0,
    "utterances": [
        {"speaker": "agent", "start_time": 0.0, "end_time": 2.0, "text": "Çağrı merkezimize hoş geldiniz, adım Ayşe. Size nasıl yardımcı olabilirim?"},
        {"speaker": "customer", "start_time": 2.5, "end_time": 8.0, "text": "Merhaba, internet faturam bu ay normalin iki katı gelmiş. Ben bu kadar kullanım yapmadım, iptal etmek istiyorum bu ne saçmalıktır!"},
        {"speaker": "agent", "start_time": 8.5, "end_time": 12.0, "text": "Yaşadığınız sorun için çok üzgünüm efendim. Hemen faturanızı kontrol ediyorum, lütfen hatta kalın."}
    ]
}

print("Analiz başlatılıyor...")
print("-" * 50)
result = get_analysis(sample_stt)
print("-" * 50)
print("Analiz Sonucu:")
print(json.dumps(result, indent=2, ensure_ascii=False))
