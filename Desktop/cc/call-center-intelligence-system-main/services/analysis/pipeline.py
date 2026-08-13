import os
import re
import json
import logging
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
logger = logging.getLogger(__name__)
_model = None
_tokenizer = None
BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "models", "llama3.2_3b_callcenter_model")
def _get_device() -> str:
    """Returns the best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
def _load_model():
    """Loads the base model and LoRA adapter (singleton pattern)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    device = _get_device()
    logger.info("Device: %s", device)
    warnings.filterwarnings("ignore", category=UserWarning)
    logger.info("Loading base model: %s", BASE_MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    if os.path.exists(ADAPTER_PATH):
        try:
            logger.info("Loading LoRA adapter from %s", ADAPTER_PATH)
            _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            logger.info("LoRA adapter loaded successfully")
        except Exception as e:
            logger.warning("Failed to load LoRA adapter, using base model only: %s", e)
            _model = base_model
    else:
        logger.warning("Adapter not found at %s — running with base model only", ADAPTER_PATH)
        _model = base_model
    _model.eval()
    return _model, _tokenizer
def _format_utterances(stt_output: dict) -> str:
    """Converts STT utterances to training-compatible input format.
    Training format: ``[0.0s] customer (-15.1 dB): Merhaba, ...``
    """
    lines = []
    for u in stt_output.get("utterances", []):
        speaker = u.get("speaker", "unknown")
        text = u.get("text", "").strip()
        start = u.get("start_time", u.get("start", 0))
        volume = u.get("volume_db", -15.0)
        if text:
            lines.append(f"[{start}s] {speaker} ({volume} dB): {text}")
    return "\n".join(lines)
# Label aliases used when parsing the model's structured text output.
_LABEL_MAP = {
    "özet": "summary",
    "summary": "summary",
    "genel duygu": "overall_sentiment",
    "overall_sentiment": "overall_sentiment",
    "sentiment": "overall_sentiment",
    "duygu": "overall_sentiment",
    "duygu skoru": "sentiment_score",
    "sentiment_score": "sentiment_score",
    "şikayet kategorisi": "complaint_category",
    "kategori": "complaint_category",
    "complaint_category": "complaint_category",
    "category": "complaint_category",
    "konu": "complaint_category",
    "anahtar kelimeler": "keywords",
    "keywords": "keywords",
    "müşteri duygu durumu": "customer_sentiment_detail",
    "temsilci performans skoru": "agent_performance_score",
    "agent_performance_score": "agent_performance_score",
    "performans": "agent_performance_score",
    "temsilci değerlendirmesi": "agent_evaluation",
}
_DEFAULT_RESULT = {
    "overall_sentiment": "neutral",
    "sentiment_score": 0.0,
    "complaint_category": "unknown",
    "keywords": [],
    "summary": "Analiz başarısız oldu veya model beklenen formatta yanıt vermedi.",
    "agent_performance_score": 0.5,
    "customer_sentiment_detail": "",
    "agent_evaluation": "",
}
def _parse_response(response_text: str) -> dict:
    """Extracts structured fields from the model's raw text output.
    Tries JSON extraction first; falls back to line-based label matching.
    """
    result = dict(_DEFAULT_RESULT)
    # Attempt JSON extraction
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            for key in result:
                if key in parsed:
                    result[key] = parsed[key]
            logger.info("Parsed response as JSON")
            return result
        except json.JSONDecodeError:
            pass
    # Fall back to line-based parsing with fuzzy label matching
    parsed_any = False
    for line in response_text.split("\n"):
        line = line.strip().lstrip("-•* ")
        if not line or ":" not in line:
            continue
        colon_idx = line.index(":")
        raw_label = line[:colon_idx].strip().lower().replace("_", " ")
        value = line[colon_idx + 1:].strip()
        matched_key = _LABEL_MAP.get(raw_label)
        if not matched_key:
            for candidate, key in _LABEL_MAP.items():
                if candidate in raw_label or raw_label in candidate:
                    matched_key = key
                    break
        if not matched_key or not value:
            continue
        parsed_any = True
        if matched_key == "sentiment_score":
            try:
                result["sentiment_score"] = float(value)
            except ValueError:
                pass
        elif matched_key == "agent_performance_score":
            try:
                if "/" in value:
                    num, den = value.split("/")
                    result["agent_performance_score"] = float(num.strip()) / float(den.strip())
                else:
                    score = float(value)
                    result["agent_performance_score"] = score / 10.0 if score > 1 else score
            except ValueError:
                pass
        elif matched_key == "keywords":
            result["keywords"] = [k.strip().strip("\"'") for k in value.split(",") if k.strip()]
        else:
            result[matched_key] = value
    if parsed_any:
        logger.info("Parsed response via line-based matching")
    elif response_text:
        result["summary"] = response_text[:500]
        logger.warning("No structured output found — raw text used as summary")
    return result
def get_analysis(stt_output: dict) -> dict:
    """

    Analyzes a transcription result and returns sentiment, category, and summary.
    Args:
        stt_output: Dict matching the STTOutput schema (see ``shared/contracts.py``).
    Returns:
        Dict with keys: ``overall_sentiment``, ``sentiment_score``,
        ``complaint_category``, ``keywords``, ``summary``,
        ``agent_performance_score``, ``customer_sentiment_detail``,
        ``agent_evaluation``.
    """
    model, tokenizer = _load_model()
    transcript_text = _format_utterances(stt_output)
    if not transcript_text:
        logger.warning("Empty transcription — skipping analysis")
        return dict(_DEFAULT_RESULT)

    # 1. Retrieve relevant company policies from Pinecone using RAG
    policy_context = ""
    try:
        from services.retrieval.rag_module import retrieve_context
        
        # Use only customer utterances for the RAG query to improve semantic matching
        # and prevent embedding truncation.
        customer_lines = [u for u in stt_output.get("utterances", []) if u.get("speaker") == "customer"]
        query = " ".join([u.get("text", "") for u in customer_lines])[:400]
        if not query.strip():
            query = transcript_text[:400] # Fallback if no customer utterances
            
        policy_context = retrieve_context(query)
        if policy_context:
            logger.info("RAG successfully retrieved relevant policy context.")
            print("\n" + "="*50)
            print("[RAG SUCCESS] Pinecone'dan İlgili Şirket Politikaları Çekildi:")
            print(policy_context)
            print("="*50 + "\n")
        else:
            logger.warning("RAG retrieved empty context.")
            print("\n[RAG WARNING] Pinecone'dan eşleşen bir politika bulunamadı.\n")
    except Exception as e:
        logger.error("Failed to retrieve RAG context: %s", e)
        print(f"\n[RAG ERROR] Pinecone araması sırasında hata oluştu: {e}\n")

    instruction = (
        "Sen profesyonel bir çağrı merkezi kalite analisti yapay zekasın. "
        "Aşağıdaki saniye ve ses desibeli (dB) verileriyle verilmiş çağrı dökümünü analiz et. "
        "Eğer varsa aşağıda verilen şirket politikalarını ve kurallarını referans alarak analizi gerçekleştir."
    )

    if policy_context:
        prompt = (
            f"### Talimat:\n{instruction}\n\n"
            f"### Giriş:\n"
            f"[Şirket Politikaları ve Kuralları]\n"
            f"{policy_context}\n\n"
            f"[Çağrı Dökümü]\n"
            f"{transcript_text}\n\n"
            f"### Yanıt:\n"
        )
    else:
        prompt = (
            f"### Talimat:\n{instruction}\n\n"
            f"### Giriş:\n{transcript_text}\n\n"
            f"### Yanıt:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    logger.debug("Raw model output:\n%s", response_text)
    result = _parse_response(response_text)
    logger.info(
        "Analysis complete — sentiment=%s, category=%s",
        result["overall_sentiment"],
        result["complaint_category"],
    )
    return result