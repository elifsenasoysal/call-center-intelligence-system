import os
import json
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Global değişkenler (Singleton mantığıyla modeli sadece 1 kez yüklemek için)
_model = None
_tokenizer = None

# Base model: Llama-3.2-3B Instruct. (Eğitimde 4-bit bnb kullanıldı, ancak Mac'te mps için standart modeli yüklüyoruz)
BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "models", "llama3.2_3b_callcenter_model")

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print("Model ve tokenizer yükleniyor... (Bu işlem biraz zaman alabilir)")
    
    # Mac için device belirleme (Apple Silicon için mps)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        
    print(f"Kullanılan cihaz (device): {device}")
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. Base Modeli Yükle
    print(f"Base model indiriliyor/yükleniyor: {BASE_MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True
    )

    # 2. Eğitilmiş LoRA Adaptörünü Üzerine Ekle
    if os.path.exists(ADAPTER_PATH):
        try:
            print("LoRA adaptörü yükleniyor...")
            _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            print("LoRA adaptörü başarıyla yüklendi!")
        except Exception as e:
            print(f"Uyarı: LoRA adaptörü yüklenemedi. Sadece base model kullanılacak. Hata: {e}")
            _model = base_model
    else:
        print(f"Uyarı: {ADAPTER_PATH} klasörü bulunamadı. Lütfen modeli buraya kopyaladığınızdan emin olun. Sadece base model çalışacak.")
        _model = base_model

    _model.eval()
    return _model, _tokenizer

def get_analysis(stt_output: dict) -> dict:
    """
    Analyzes a transcription result and returns sentiment, category, and summary.

    Args:
        stt_output: dict matching the STTOutput schema (see shared/contracts.py).

    Returns:
        dict with keys: overall_sentiment, sentiment_score, complaint_category,
        keywords, summary, agent_performance_score.
        See shared/mock_data/sample_llm_output.json for reference.
    """
    model, tokenizer = _load_model()
    
    # 1. Konuşmaları (Utterances) Metne Dönüştür
    utterances = stt_output.get("utterances", [])
    dialogue_lines = []
    for u in utterances:
        speaker_map = {"agent": "Müşteri Temsilcisi", "customer": "Müşteri", "unknown": "Bilinmeyen"}
        speaker_name = speaker_map.get(u.get("speaker", "unknown"), "Bilinmeyen")
        text = u.get("text", "").strip()
        if text:
            dialogue_lines.append(f"{speaker_name}: {text}")
            
    transcript_text = "\n".join(dialogue_lines)
    
    # Eğer metin boşsa analize gerek yok
    if not transcript_text:
        return {}
    
    # 2. Prompt'u Hazırla (Eğitimdeki formata tam uyumlu)
    instruction = (
        "Aşağıdaki çağrı dökümünü analiz et ve bir JSON formatında şu bilgileri çıkar: "
        "overall_sentiment, sentiment_score, complaint_category, keywords, summary, ve agent_performance_score."
    )
    
    prompt = (
        f"### Talimat:\n{instruction}\n\n"
        f"### Giriş:\n{transcript_text}\n\n"
        f"### Yanıt:\n"
    )
    
    # 3. Model Çıkarımı
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1,  # JSON üretiminde tutarlılık için düşük sıcaklık
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        
    # Sadece yeni üretilen tokenları al
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 4. JSON Ayrıştırma (Eğer model başına sonuna markdown backtick koyduysa temizle)
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
        
    try:
        result_dict = json.loads(response_text)
        return result_dict
    except json.JSONDecodeError:
        print(f"JSON Ayrıştırma Hatası. Model çıktısı:\n{response_text}")
        # Hata durumunda yedek (fallback) değerler dön
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "complaint_category": "unknown",
            "keywords": [],
            "summary": "Analiz başarısız oldu veya model beklenen JSON formatında yanıt vermedi.",
            "agent_performance_score": 0.5
        }
