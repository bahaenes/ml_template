# Makine Öğrenmesi Proje Şablonu

Bu şablon, makine öğrenmesi projelerini hızlı ve düzenli bir şekilde geliştirmek için tasarlanmıştır. Veri analizi, önişleme, model eğitimi ve değerlendirme adımlarını içerir.

## Özellikler

- Otomatik veri yükleme ve analiz
- Kapsamlı keşifsel veri analizi (EDA)
- Otomatik veri önişleme
- Çoklu model eğitimi ve karşılaştırma
- Detaylı model değerlendirme ve görselleştirme
- Sonuçların ve modellerin otomatik kaydedilmesi
- Yapılandırılabilir proje ayarları

## Kurulum

1. Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setinizi `data` klasörüne yerleştirin.

3. `config.yaml` dosyasını projenize göre düzenleyin:
   - Veri seti yolunu belirtin
   - Problem türünü seçin (classification/regression)
   - Hedef değişkeni belirtin
   - Model parametrelerini ayarlayın

## Kullanım

1. Projeyi çalıştırın:
```bash
python ml_project_template.py
```

2. Sonuçlar `outputs` klasöründe oluşturulacaktır:
   - `eda/`: Veri analizi grafikleri ve istatistikler
   - `model_evaluation/`: Model performans grafikleri
   - `model/`: Eğitilmiş en iyi model ve sonuçlar

## Proje Yapısı

```
.
├── data/                    # Veri setleri
├── outputs/                 # Çıktılar
│   ├── eda/                # Veri analizi sonuçları
│   ├── model_evaluation/   # Model değerlendirme sonuçları
│   └── model/             # Eğitilmiş modeller
├── config.yaml             # Proje yapılandırması
├── requirements.txt        # Bağımlılıklar
├── README.md              # Dokümantasyon
└── ml_project_template.py # Ana kod
```

## Özelleştirme

1. Yeni modeller eklemek için `config.yaml` dosyasındaki `models` bölümünü güncelleyin.
2. Özel önişleme adımları eklemek için `preprocess_data` metodunu düzenleyin.
3. Yeni metrikler eklemek için `_calculate_metrics` metodunu güncelleyin.
4. Özel görselleştirmeler için `_plot_model_evaluation` metodunu düzenleyin.

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun 