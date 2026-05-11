import sys
import os
import streamlit as st
import tempfile

# Proje kök dizinini sys.path'e ekleyelim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json

# Şimdilik torch/whisper bağımlılıklarından kaçınmak için import'u yoruma alıyoruz
# from services.transcription.pipeline import get_transcription
from services.analysis.pipeline import get_analysis

def mock_get_transcription(file_path):
    import time
    time.sleep(2) # Fake processing time
    mock_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'mock_data', 'sample_stt_output.json')
    with open(mock_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

st.set_page_config(
    page_title="Call Center Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .positive { color: #2ca02c; }
    .negative { color: #d62728; }
    .neutral { color: #ff7f0e; }
</style>
""", unsafe_allow_html=True)

st.title("🎧 Call Center Intelligence Dashboard")
st.markdown("Yapay zeka destekli çağrı analizi, otomatik deşifre ve duygu analizi sistemi.")

# Sidebar
st.sidebar.header("Kontrol Paneli")
st.sidebar.info("Lütfen analiz etmek istediğiniz müşteri görüşmesi ses kaydını yükleyin.")

uploaded_file = st.sidebar.file_uploader("Ses Dosyası Yükle", type=["wav", "mp3", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    st.sidebar.success("Dosya başarıyla yüklendi!")
    st.audio(uploaded_file)

    if st.button("🚀 Analizi Başlat", type="primary"):
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        try:
            # 1. STT (Transcription)
            with st.spinner("Ses dosyası metne dönüştürülüyor (STT)..."):
                stt_result = mock_get_transcription(temp_file_path)
            
            # 2. LLM Analysis
            with st.spinner("Metin yapay zeka ile analiz ediliyor (LLM)..."):
                analysis_result = get_analysis(stt_result)
            
            st.success("Analiz tamamlandı!")

            st.divider()

            # Dashboard Layout
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("📊 Analiz Özeti")
                
                # Sentiment Display
                sentiment = analysis_result.get('overall_sentiment', 'neutral')
                sentiment_color = "positive" if sentiment == "positive" else "negative" if sentiment == "negative" else "neutral"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Duygu Durumu (Sentiment)</div>
                    <div class="metric-value {sentiment_color}">{sentiment.capitalize()}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("") # Spacer
                
                # Category Display
                category = analysis_result.get('complaint_category', 'Bilinmiyor')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Şikayet Kategorisi</div>
                    <div class="metric-value">{category.capitalize()}</div>
                </div>
                """, unsafe_allow_html=True)

                st.write("") # Spacer
                
                # Agent Score Display
                score = analysis_result.get('agent_performance_score', 'N/A')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Temsilci Performans Skoru</div>
                    <div class="metric-value">{score} / 1.0</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.subheader("📝 Çağrı Özeti")
                st.info(analysis_result.get('summary', 'Özet bulunamadı.'))
                
                st.subheader("🔑 Anahtar Kelimeler")
                keywords = analysis_result.get('keywords', [])
                if keywords:
                    st.write(", ".join([f"`{kw}`" for kw in keywords]))
                else:
                    st.write("Anahtar kelime bulunamadı.")

            st.divider()

            # Transcription Display
            st.subheader("💬 Görüşme Dökümü (Transkripsiyon)")
            with st.expander("Metnin tamamını görmek için tıklayın", expanded=True):
                if 'utterances' in stt_result:
                    for utt in stt_result['utterances']:
                        speaker = utt.get('speaker', 'Unknown')
                        text = utt.get('text', '')
                        start = utt.get('start', 0)
                        end = utt.get('end', 0)
                        
                        # Style differently based on speaker if possible
                        st.markdown(f"**[{start:.2f}s - {end:.2f}s] {speaker}:** {text}")
                else:
                    st.warning("Metin dökümü bulunamadı veya format hatalı.")

        except Exception as e:
            st.error(f"Analiz sırasında bir hata oluştu: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
else:
    st.info("👆 Analizi başlatmak için soldaki menüden bir ses dosyası yükleyin.")
