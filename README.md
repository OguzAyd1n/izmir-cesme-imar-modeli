# İzmir/Çeşme İmara Açılabilecek Arsa Tespit Modeli

Bu proje, İzmir/Çeşme bölgesinde imara açılabilecek arsaların tespiti için geliştirilmiş bir makine öğrenmesi modelidir.

## Özellikler

- Kadastro, imar planı ve altyapı verilerinin entegrasyonu
- Kural tabanlı filtreleme sistemi
- Makine öğrenmesi modeli
- Hibrit puanlama sistemi
- Görselleştirme ve raporlama araçları

## Gereksinimler

- Python 3.8+
- pandas
- numpy
- geopandas
- rasterio
- shapely
- scikit-learn
- matplotlib
- seaborn
- folium

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri klasörünü oluşturun:
```bash
mkdir data
```

3. Gerekli veri dosyalarını `data` klasörüne yerleştirin:
- Kadastro verileri
- İmar planı verileri
- Altyapı verileri
- Topografya verileri
- Jeoloji verileri
- Koruma alanları verileri

## Kullanım

Modeli çalıştırmak için:

```python
python izmir_cesme_imar_modeli.py
```

## Çıktılar

Model çalıştırıldığında aşağıdaki çıktılar oluşturulur:
- İmar uygunluk haritası
- Kategori sınıflandırma haritası
- İnteraktif harita
- İstatistiksel grafikler
- CSV raporu

## Lisans

MIT

## İletişim

- GitHub: [OguzAyd1n](https://github.com/OguzAyd1n) 