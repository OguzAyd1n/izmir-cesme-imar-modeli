import geopandas as gpd
import rasterio
import numpy as np
from rasterio.plot import show
import matplotlib
matplotlib.use('Agg')  # Backend'i Agg olarak ayarla
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon
import os

def imar_uygunluk_analizi():
    """İmara uygun alanları belirle"""
    print("İmar uygunluk analizi başladı...")
    
    try:
        # DEM verisini oku
        print("DEM verisi okunuyor...")
        with rasterio.open("data/topografya/cesme_dem.tif") as src:
            dem = src.read(1)
            transform = src.transform
            
            # Eğim hesapla
            print("Eğim hesaplanıyor...")
            dx = np.gradient(dem, axis=1)
            dy = np.gradient(dem, axis=0)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            
            # Eğim uygunluk skoru (0-1 arası)
            slope_score = np.where(
                slope < 5, 1.0,  # Düz alanlar: 1
                np.where(
                    slope < 15, 0.8,  # Hafif eğimli: 0.8
                    np.where(
                        slope < 25, 0.5,  # Orta eğimli: 0.5
                        np.where(
                            slope < 45, 0.2,  # Dik: 0.2
                            0.0  # Çok dik: 0
                        )
                    )
                )
            )
            
            # Yükseklik uygunluk skoru (0-1 arası)
            height_score = np.where(
                (dem > 0) & (dem < 100), 1.0,  # 0-100m arası: 1
                np.where(
                    (dem >= 100) & (dem < 200), 0.8,  # 100-200m arası: 0.8
                    np.where(
                        (dem >= 200) & (dem < 300), 0.5,  # 200-300m arası: 0.5
                        np.where(
                            (dem >= 300) & (dem < 400), 0.2,  # 300-400m arası: 0.2
                            0.0  # 400m üstü: 0
                        )
                    )
                )
            )
            
            # Arazi kullanım verilerini oku
            print("Arazi kullanım verileri okunuyor...")
            zoning = gpd.read_file("data/imar/cesme_zoning.shp")
            
            # Arazi kullanım uygunluk skorları (imar açısından)
            landuse_scores = {
                'residential': 1.0,  # Mevcut yerleşim alanları
                'commercial': 1.0,   # Ticari alanlar
                'industrial': 0.9,   # Endüstriyel alanlar
                'farmland': 0.8,     # Tarım arazileri
                'forest': 0.3,       # Orman alanları
                'water': 0.0,        # Su alanları
                'wetland': 0.0,      # Sulak alanlar
                'quarry': 0.7,       # Taş ocakları
                'construction': 1.0,  # İnşaat alanları
                'grass': 0.8,        # Çayır alanları
                'meadow': 0.8,       # Mera alanları
                'scrub': 0.7,        # Çalılık alanlar
                'heath': 0.6,        # Fundalık alanlar
                'bare_rock': 0.5,    # Çıplak kayalık
                'sand': 0.4,         # Kum alanları
                'mud': 0.0,          # Bataklık
                'scree': 0.3,        # Moloz alanları
                'bay': 0.0,          # Koy
                'beach': 0.0,        # Plaj
                'coastline': 0.0,    # Kıyı şeridi
                'reservoir': 0.0,    # Rezervuar
                'dam': 0.0,          # Baraj
                'harbour': 0.0,      # Liman
                'apartments': 1.0,   # Apartmanlar
                'house': 1.0,        # Evler
                'semidetached_house': 1.0,  # Müstakil evler
                'hotel': 1.0,        # Oteller
                'school': 1.0,       # Okullar
                'mosque': 1.0,       # Camiler
                'public': 1.0,       # Kamu alanları
                'retail': 1.0,       # Perakende
                'warehouse': 0.9,    # Depolar
                'greenhouse': 0.8,   # Seralar
                'greenhouse_horticulture': 0.8,  # Bahçecilik
                'orchard': 0.7,      # Meyve bahçeleri
                'vineyard': 0.7,     # Bağlar
                'farmyard': 0.7,     # Çiftlik alanları
                'wine_press_house': 0.7,  # Şarap evleri
                'cemetery': 0.5,     # Mezarlıklar
                'ruins': 0.6,        # Harabeler
                'abandoned': 0.6,    # Terkedilmiş alanlar
                'yes': 0.5           # Diğer
            }
            
            # Rasterize edilmiş arazi kullanım uygunluk skoru
            print("Arazi kullanım skorları hesaplanıyor...")
            landuse_score = np.zeros_like(dem, dtype=float)
            
            # Her arazi kullanım poligonu için
            for idx, row in zoning.iterrows():
                # Poligonun sınırlarını al
                bounds = row.geometry.bounds
                
                # Raster koordinatlarına dönüştür
                col_start, row_start = ~transform * (bounds[0], bounds[3])
                col_end, row_end = ~transform * (bounds[2], bounds[1])
                
                # Piksel indekslerini hesapla
                col_start, row_start = int(col_start), int(row_start)
                col_end, row_end = int(col_end), int(row_end)
                
                # Geçerli indeksler
                col_start = max(0, min(col_start, dem.shape[1]))
                col_end = max(0, min(col_end, dem.shape[1]))
                row_start = max(0, min(row_start, dem.shape[0]))
                row_end = max(0, min(row_end, dem.shape[0]))
                
                # Uygunluk skorunu ata
                landuse_score[row_start:row_end, col_start:col_end] = landuse_scores.get(row['FONKSIYON'], 0.5)
            
            # Toplam uygunluk skoru (0-1 arası)
            print("Toplam uygunluk skoru hesaplanıyor...")
            # Ağırlıklar: Eğim %40, Yükseklik %30, Arazi Kullanımı %30
            total_score = (slope_score * 0.4 + height_score * 0.3 + landuse_score * 0.3)
            
            # İmar uygunluk sınıfları
            imar_classes = {
                'İmara Çok Uygun': np.sum(total_score >= 0.8),
                'İmara Uygun': np.sum((total_score >= 0.6) & (total_score < 0.8)),
                'İmara Orta Uygun': np.sum((total_score >= 0.4) & (total_score < 0.6)),
                'İmara Az Uygun': np.sum((total_score >= 0.2) & (total_score < 0.4)),
                'İmara Uygun Değil': np.sum(total_score < 0.2)
            }
            
            print("\nİmar Uygunluk Sınıfları:")
            for sinif, piksel in imar_classes.items():
                yuzde = (piksel / total_score.size) * 100
                print(f"{sinif}: {yuzde:.2f}%")
            
            # İmara uygun alanları vektör veriye dönüştür
            print("\nİmara uygun alanlar belirleniyor...")
            imar_uygun = total_score >= 0.6  # İmara uygun ve çok uygun alanlar
            
            # Bağlantılı bileşenleri bul
            from scipy.ndimage import label
            labeled_array, num_features = label(imar_uygun)
            print(f"Toplam {num_features} adet bağlantılı alan bulundu.")
            
            # Her bileşen için alan hesapla
            min_alan = 10000  # Minimum 1 hektar
            uygun_alanlar = []
            
            for i in range(1, num_features + 1):
                # Bileşenin piksel sayısını bul
                piksel_sayisi = np.sum(labeled_array == i)
                
                # Piksel alanını hesapla (m²)
                piksel_alan = abs(transform[0] * transform[4])
                alan = piksel_sayisi * piksel_alan
                
                # Minimum alan kontrolü
                if alan >= min_alan:
                    # Bileşenin sınırlarını bul
                    rows, cols = np.where(labeled_array == i)
                    if len(rows) > 0 and len(cols) > 0:
                        # Raster koordinatlarını gerçek koordinatlara dönüştür
                        x_coords = transform[0] * cols + transform[2]
                        y_coords = transform[4] * rows + transform[5]
                        
                        # Poligon oluştur
                        coords = list(zip(x_coords, y_coords))
                        if len(coords) >= 3:  # En az 3 nokta gerekli
                            poligon = Polygon(coords)
                            uygun_alanlar.append({
                                'geometry': poligon,
                                'alan': alan / 10000,  # hektar cinsinden
                                'uygunluk': np.mean(total_score[labeled_array == i])
                            })
            
            # GeoDataFrame oluştur
            if uygun_alanlar:
                uygun_gdf = gpd.GeoDataFrame(uygun_alanlar, geometry='geometry')
                uygun_gdf.crs = src.crs
                
                # Kaydet
                print(f"\n{len(uygun_alanlar)} adet imara uygun alan bulundu ve kaydediliyor...")
                uygun_gdf.to_file("data/analiz/imar_uygun_alanlar.shp")
                print("İmara uygun alanlar kaydedildi.")
                
                # Görselleştirme
                print("\nGörselleştirme oluşturuluyor...")
                plt.figure(figsize=(10, 10))
                uygun_gdf.plot(column='uygunluk', cmap='RdYlGn', legend=True)
                plt.title('İmara Uygun Alanlar')
                plt.savefig("data/analiz/imar_uygun_alanlar.png", dpi=300, bbox_inches='tight')
                print("Görselleştirme kaydedildi.")
            
    except Exception as e:
        print(f"İmar uygunluk analizi sırasında hata: {str(e)}")

def main():
    """Ana fonksiyon"""
    # Analiz klasörünü oluştur
    os.makedirs("data/analiz", exist_ok=True)
    
    # İmar uygunluk analizini yap
    imar_uygunluk_analizi()
    
    print("\nİmar uygunluk analizi tamamlandı. Sonuçlar data/analiz klasöründe kaydedildi.")

if __name__ == "__main__":
    main() 