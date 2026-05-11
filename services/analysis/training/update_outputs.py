"""
train_data.jsonl dosyasındaki placeholder output'ları
gerçek analiz metinleriyle günceller.
"""
import json
import os

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH   = os.path.join(TRAINING_DIR, "train_data.jsonl")

# 11 çağrı için gerçek analiz output'ları (sırasıyla call_001 … call_011)
REAL_OUTPUTS = [

    # --- call_001: Harvey - veri hızı yavaş ---
    """Özet: Müşteri Harvey, video izleme ve büyük dosya indirirken veri hızlarının yavaşladığını bildirdi. Temsilci Ella, hesabı inceleyerek müşterinin kapasitesi dolu bir hücreye bağlı olduğunu tespit etti ve öncelikli veri sağlayan yeni bir plana geçişi anında tamamladı. Müşteri sorunun çözüldüğünü öğrenerek memnuniyetle ayrıldı.
Genel Duygu: pozitif
Duygu Skoru: 0.7
Şikayet Kategorisi: teknik sorun - veri hızı
Anahtar Kelimeler: veri hızı, yavaş bağlantı, plan yükseltme, hücre kapasitesi
Müşteri Duygu Durumu: Başlangıçta endişeli; çözüm sonrasında memnun ve teşekkür eden
Temsilci Performans Skoru: 9/10
Temsilci Değerlendirmesi: Sorunu hızla teşhis etti, uygun planı önerdi ve işlemi anında tamamladı. Profesyonel ve yardımsever tutum sergiledi.""",

    # --- call_002: Kirsten - otomatik ödeme güvenliği ---
    """Özet: Müşteri Kirsten, otomatik fatura ödeme sisteminin güvenlik önlemlerini sorgulamak için aradı. Temsilci kimlik doğrulaması için PIN veya kredi kartı bilgisi talep etti; müşteri bu bilgileri telefonda paylaşmaktan kaçındı. Temsilci alternatif olarak web sitesi üzerinden giriş yapılmasını önerdi ve web adresini iletti.
Genel Duygu: nötr
Duygu Skoru: 0.1
Şikayet Kategorisi: fatura güvenliği - kimlik doğrulama
Anahtar Kelimeler: otomatik ödeme, güvenlik, PIN, kimlik doğrulama, web sitesi
Müşteri Duygu Durumu: Temkinli; hassas bilgi paylaşmak istemedi ancak önerilen çözümü kabul etti
Temsilci Performans Skoru: 7/10
Temsilci Değerlendirmesi: Alternatif çözüm sundu ancak güvenlik protokolünü daha ayrıntılı açıklayabilirdi.""",

    # --- call_003: Dora - şüpheli hesap aktivitesi ---
    """Özet: Müşteri Dora, hesabında tuhaf mesajlar ve aramalar aldığını, hesabının ele geçirilmiş olabileceğinden endişelendiğini bildirdi. Temsilci Dan hesapta olağandışı işlemler tespit ederek güvenlik ekibini devreye aldı. Müşteriye yeni şifre oluşturması ve iki faktörlü doğrulamayı etkinleştirmesi tavsiye edildi.
Genel Duygu: negatif
Duygu Skoru: -0.4
Şikayet Kategorisi: güvenlik - hesap ihlali şüphesi
Anahtar Kelimeler: şüpheli aktivite, hesap güvenliği, iki faktörlü doğrulama, şifre değiştirme
Müşteri Duygu Durumu: Endişeli ve tedirgin; durumun ciddiyetini kavradı ancak tam olarak rahatlamadı
Temsilci Performans Skoru: 8/10
Temsilci Değerlendirmesi: Güvenlik önlemlerini açıkladı ve güvenlik ekibini devreye aldı. Müşteriyi daha fazla sakinleştirebilirdi.""",

    # --- call_004: Mike - uluslararası roaming ---
    """Özet: Müşteri Mike, yurtdışında bulunduğu sırada veri ve arama hizmetlerine erişemediğini bildirdi. Temel sorun giderme adımlarını denemiş ancak çözüm bulamamıştı. Temsilci Abbigail, hesaptaki uluslararası roaming ayarlarında hata tespit ederek düzeltti. Müşteri sorunun çözüldüğünü öğrenerek teşekkür etti.
Genel Duygu: pozitif
Duygu Skoru: 0.8
Şikayet Kategorisi: teknik sorun - uluslararası roaming
Anahtar Kelimeler: roaming, yurtdışı, veri erişimi, bağlantı sorunu, hesap ayarları
Müşteri Duygu Durumu: Başlangıçta endişeli ve çözüm arayan; sorun çözüldükten sonra son derece memnun
Temsilci Performans Skoru: 9/10
Temsilci Değerlendirmesi: Sorunu hızla tespit edip çözdü. Ek destek teklif etti. Hizmet kalitesi yüksekti.""",

    # --- call_005: Besse - uygulama hatası ---
    """Özet: Müşteri Besse, telefonundaki üçüncü taraf bir uygulamada özelliklere erişemediğini ve hata mesajları aldığını bildirdi. Temsilci Becky, kimlik doğrulamasını tamamladıktan sonra sorunun mobil hizmetten değil uygulamanın kendisinden kaynaklandığını belirleyerek müşteriyi uygulamanın destek ekibine yönlendirdi; ancak destek iletişim bilgilerini sağlayamadı.
Genel Duygu: nötr
Duygu Skoru: -0.1
Şikayet Kategorisi: teknik sorun - üçüncü taraf uygulama hatası
Anahtar Kelimeler: uygulama hatası, erişim sorunu, hata mesajı, destek yönlendirme
Müşteri Duygu Durumu: Hafif hayal kırıklığı; çözüm alamadan başka bir yere yönlendirilmek zorunda kaldı
Temsilci Performans Skoru: 6/10
Temsilci Değerlendirmesi: Sorunun kaynağını doğru tespit etti ancak uygulamanın destek bilgilerini sağlayamaması eksiklik oluşturdu.""",

    # --- call_006: Eugenie - plan değişikliği ---
    """Özet: Müşteri Eugenie, mevcut planını değiştirmek istedi. Temsilci Ezra, ihtiyaçlarını analiz ederek aylık 40 dolar karşılığında 500 MB veri, 500 dakika konuşma ve 500 SMS içeren Single Line Plan 500'i önerdi. Plan değişikliği tamamlandı; ilk üç ay için ücretsiz Çağrı İade özelliği eklendi.
Genel Duygu: pozitif
Duygu Skoru: 0.8
Şikayet Kategorisi: plan değişikliği - yükseltme talebi
Anahtar Kelimeler: plan değişikliği, Single Line Plan 500, çağrı iade, aylık ücret, veri paketi
Müşteri Duygu Durumu: Memnun; ihtiyaçlarına uygun plan seçildi ve ek özellik kazanıldı
Temsilci Performans Skoru: 9/10
Temsilci Değerlendirmesi: Müşteri ihtiyaçlarını analiz ederek uygun planı önerdi. Ek özellik sunarak değer kattı. Satış odaklı ancak müşteri yararına hizmet etti.""",

    # --- call_007: Breanne - yazılım güncelleme ---
    """Özet: Müşteri Breanne, cihazının yazılım güncellemesinde sorun yaşadığını bildirdi. Temsilci Geoffrey, bu konuda yardımcı olamayacağını belirterek müşteriyi ayrı bir destek hattına (1-800-123-4567) yönlendirdi. Müşteri hayal kırıklığını belirtti ancak bilgiyi not alarak çağrıyı kapattı.
Genel Duygu: negatif
Duygu Skoru: -0.5
Şikayet Kategorisi: teknik sorun - yazılım güncelleme
Anahtar Kelimeler: yazılım güncelleme, cihaz sorunu, destek hattı, yönlendirme, çözümsüz çağrı
Müşteri Duygu Durumu: Hayal kırıklığı; sorununu çözdüremeden başka bir hatta yönlendirildi
Temsilci Performans Skoru: 5/10
Temsilci Değerlendirmesi: Doğru hatta yönlendirdi ancak empati kurmadan çağrıyı kapattı. Müşteriye daha iyi destek sağlanabilirdi.""",

    # --- call_008: Wilmer - otomatik ödeme doğrulama ---
    """Özet: Müşteri Wilmer, otomatik fatura ödeme ayarlarının doğru yapılandırıldığını teyit ettirmek için aradı. Temsilci Keisha, hesap numarasını alarak inceleme yaptı ve ödemelerin sorunsuz işlendiğini doğrulayarak müşteriyi bilgilendirdi. Müşteri memnuniyetle ayrıldı.
Genel Duygu: pozitif
Duygu Skoru: 0.9
Şikayet Kategorisi: fatura - otomatik ödeme doğrulama
Anahtar Kelimeler: otomatik ödeme, fatura kontrolü, hesap doğrulama, ödeme onayı
Müşteri Duygu Durumu: Sakin ve merak eden; bilgiyi aldıktan sonra tamamen memnun
Temsilci Performans Skoru: 9/10
Temsilci Değerlendirmesi: Hızlı ve etkin şekilde bilgi sağladı. Müşteri talebi net karşılandı. Profesyonel tutum sergilendi.""",

    # --- call_009: Selina - add-on iptali ---
    """Özet: Müşteri Selina, aylar önce iptal etmeye çalıştığı ancak hâlâ aktif olan bir eklentiden duyduğu rahatsızlığı dile getirdi. Temsilci Dominic kimlik doğrulamasını tamamlayarak hesaptaki sorunu tespit etti ve add-on'u anında iptal etti. Müşteriye onay mesajı gönderildi.
Genel Duygu: pozitif
Duygu Skoru: 0.5
Şikayet Kategorisi: fatura - hatalı eklenti iptali
Anahtar Kelimeler: add-on, eklenti iptali, hatalı faturalandırma, hesap düzeltme, onay mesajı
Müşteri Duygu Durumu: Başlangıçta sinirli ve hayal kırıklığı içinde; sorun çözüldükten sonra memnun ve teşekkür eden
Temsilci Performans Skoru: 9/10
Temsilci Değerlendirmesi: Müşterinin şikayetini ciddiye alarak hızla çözdü. Empati kurdu ve işlemi anında tamamladı.""",

    # --- call_010: Gwen - 5G bağlantı kesintisi ---
    """Özet: Müşteri Gwen, iyi kapsama alanında olmasına rağmen 5G cihazında internet bağlantısı kuramadığını bildirdi. Yeniden başlatma ve güncelleme kontrolü yapmıştı ancak sorun devam ediyordu. Temsilci Reginald temel sorun giderme adımlarını sorguladı; çözüm bulamayarak müşteriyi teknik destek ekibine transfer etti.
Genel Duygu: negatif
Duygu Skoru: -0.3
Şikayet Kategorisi: teknik sorun - 5G internet bağlantı kesintisi
Anahtar Kelimeler: 5G bağlantısı, internet kesintisi, teknik destek, transfer, sinyal sorunu
Müşteri Duygu Durumu: Endişeli ve çözüm arayan; transfer sonrasında belirsizlik içinde
Temsilci Performans Skoru: 6/10
Temsilci Değerlendirmesi: Temel sorun giderme adımlarını uyguladı ancak çözüm üretemedi. Transfer kararı doğruydu; müşteriyi transfer öncesinde daha iyi bilgilendirmeliydi.""",

    # --- call_011: Karin - kurumsal veri planı ---
    """Özet: Müşteri, işletmesi için mobil veri planı ve hotspot hizmetleri hakkında bilgi almak istedi. Temsilci Maryann, 10 cihaz desteği, yüksek hızlı veri, 24/7 teknik destek, uluslararası roaming ve güvenlik özellikleri içeren kurumsal planı ayrıntılı olarak anlattı. Müşteri ilgi gösterdi; temsilci bir sonraki Çarşamba için geri arama planladı.
Genel Duygu: pozitif
Duygu Skoru: 0.6
Şikayet Kategorisi: satış - kurumsal plan bilgi talebi
Anahtar Kelimeler: kurumsal plan, mobil hotspot, veri paketi, uluslararası roaming, geri arama
Müşteri Duygu Durumu: Meraklı ve değerlendirme yapan; plan detaylarına ilgi gösterdi ve sonraki adım için hazır
Temsilci Performans Skoru: 8/10
Temsilci Değerlendirmesi: Kurumsal ihtiyaçları anlayarak kapsamlı bilgi sundu. Geri arama planlaması ile satış sürecini ilerletti. Daha hedefli sorularla müşteriyi daha iyi yönlendirebilirdi.""",
]

def main():
    # Mevcut JSONL satırlarını oku
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if len(lines) != len(REAL_OUTPUTS):
        print(f"⚠️  Uyarı: JSONL'de {len(lines)} satır var, "
              f"ama {len(REAL_OUTPUTS)} output tanımlı.")
        print("Eşleşen satırlar güncellenecek, diğerleri atlanacak.")

    # Output'ları güncelle
    updated = 0
    for i, line in enumerate(lines):
        if i < len(REAL_OUTPUTS):
            line["output"] = REAL_OUTPUTS[i].strip()
            updated += 1

    # Geri yaz
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✅ {updated} satır başarıyla güncellendi → {JSONL_PATH}")

if __name__ == "__main__":
    main()
