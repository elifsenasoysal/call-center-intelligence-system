# 🧠 Model Eğitimi — Çalıştırma Rehberi

Bu klasördeki `train.py` scripti, Llama-3.2-3B modelini Türkçe çağrı merkezi verileriyle
ince ayar (fine-tuning) yaparak özel bir analiz modeli oluşturur.

---

## ⚠️ Gereksinimler

| Gereksinim | Açıklama |
|------------|----------|
| **CUDA GPU** | NVIDIA ekran kartı zorunlu (en az 16 GB VRAM önerilir). Mac ve Intel CPU ile çalışmaz. |
| **Python** | 3.10 veya üzeri |
| **İnternet** | Model ilk çalıştırmada HuggingFace'den (~5 GB) indirilir |
| **HuggingFace Hesabı** | `pyannote` modeli için erişim token'ı gerekli |

---

## 🚀 Adım Adım Kurulum ve Çalıştırma

### 1. Projeyi İndir / Klonla

```bash
git clone <repo-url>
cd call-center-intelligence-system
```

Ya da projeyi ZIP olarak kopyalarsan, klasörün içine gir:

```bash
cd call-center-intelligence-system
```

---

### 2. Python Sanal Ortam Oluştur (önerilir)

```bash
python -m venv venv
source venv/bin/activate        # Windows'ta: venv\Scripts\activate
```

---

### 3. Bağımlılıkları Kur

> ⚠️ `unsloth` paketi CUDA sürümüne göre kurulmalıdır. Önce CUDA versiyonunu kontrol et:
> ```bash
> nvidia-smi
> ```

**CUDA 12.1 için:**
```bash
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

**CUDA 11.8 için:**
```bash
pip install "unsloth[cu118-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

Ardından geri kalan paketleri kur:
```bash
pip install torch transformers trl peft bitsandbytes datasets accelerate python-dotenv
```

---

### 4. `.env` Dosyasını Oluştur

Proje **ana dizininde** (yani `call-center-intelligence-system/` klasöründe) `.env` adında bir dosya oluştur:

```
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

> HuggingFace token'ını buradan alabilirsin: https://huggingface.co/settings/tokens  
> Token'ın `Read` iznine sahip olması yeterli.

---

### 5. Eğitimi Başlat

`training/` klasörüne git ve scripti çalıştır:

```bash
cd services/analysis/training
python train.py
```

> 💡 **Not:** Script proje içindeki `train_data.jsonl` dosyasını otomatik olarak bulur,
> hangi dizinden çalıştırdığın önemli değil.

---

## ⏱️ Beklenen Süre

| Donanım | Tahmini Süre |
|---------|--------------|
| RTX 3090 / 4090 | ~15–25 dakika |
| RTX 3080 | ~25–40 dakika |
| T4 (Google Colab) | ~40–60 dakika |

---

## 📦 Eğitim Sonunda Ne Olur?

Script tamamlandığında şu klasörler oluşur:

```
services/analysis/training/
├── llama3.2_3b_callcenter_model/    ← 🎯 Eğitilmiş model burada
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer.json (+ diğer dosyalar)
└── outputs/                    ← Eğitim sırasındaki checkpoint'ler
```

> Eğitim başarıyla bittiğinde terminalde şu mesajı görürsün:
> ```
> İşlem başarıyla tamamlandı! Model şuraya kaydedildi: .../llama3.2_3b_callcenter_model
> ```

---

## ❓ Sık Karşılaşılan Hatalar

| Hata | Çözüm |
|------|-------|
| `CUDA out of memory` | `per_device_train_batch_size = 1` zaten en düşük değerde. Başka uygulamaları kapat. |
| `HUGGINGFACE_TOKEN bulunamadı` | `.env` dosyasını oluşturmayı ve token'ı eklemeyi unutmuşsun |
| `train_data.jsonl bulunamadı` | Dosyanın `training/` klasöründe olduğunu kontrol et |
| `unsloth import hatası` | Unsloth'u CUDA versiyonuna uygun şekilde yeniden kur (Adım 3) |