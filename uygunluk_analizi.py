import geopandas as gpd
import rasterio
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box
import os

def uygunluk_analizi():
    """Uygun yerleşim alanlarını belirle"""
    print("Uygunluk analizi başladı...")
    
    try:
        # DEM verisini oku
        with rasterio.open("data/topografya/cesme_dem.tif") as src:
            dem = src.read(1)
            transform = src.transform
            
            # Eğim hesapla
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
            zoning = gpd.read_file("data/imar/cesme_zoning.shp")
            
            # Arazi kullanım uygunluk skorları
            landuse_scores = {
                'residential': 1.0,
                'commercial': 0.9,
                'industrial': 0.8,
                'farmland': 0.7,
                'forest': 0.3,
                'water': 0.0,
                'wetland': 0.0,
                'quarry': 0.1,
                'construction': 0.9,
                'grass': 0.6,
                'meadow': 0.6,
                'scrub': 0.5,
                'heath': 0.4,
                'bare_rock': 0.2,
                'sand': 0.3,
                'mud': 0.0,
                'scree': 0.1,
                'bay': 0.0,
                'beach': 0.0,
                'coastline': 0.0,
                'reservoir': 0.0,
                'dam': 0.0,
                'harbour': 0.0,
                'apartments': 1.0,
                'house': 1.0,
                'semidetached_house': 1.0,
                'hotel': 0.9,
                'school': 0.9,
                'mosque': 0.9,
                'public': 0.9,
                'retail': 0.9,
                'warehouse': 0.8,
                'greenhouse': 0.7,
                'greenhouse_horticulture': 0.7,
                'orchard': 0.7,
                'vineyard': 0.7,
                'farmyard': 0.7,
                'wine_press_house': 0.7,
                'cemetery': 0.5,
                'ruins': 0.3,
                'abandoned': 0.3,
                'yes': 0.5
            }
            
            # Rasterize edilmiş arazi kullanım uygunluk skoru
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
            total_score = (slope_score * 0.4 + height_score * 0.3 + landuse_score * 0.3)
            
            # Uygunluk sınıfları
            suitability_classes = {
                'Çok Uygun': np.sum(total_score >= 0.8),
                'Uygun': np.sum((total_score >= 0.6) & (total_score < 0.8)),
                'Orta Uygun': np.sum((total_score >= 0.4) & (total_score < 0.6)),
                'Az Uygun': np.sum((total_score >= 0.2) & (total_score < 0.4)),
                'Uygun Değil': np.sum(total_score < 0.2)
            }
            
            print("\nUygunluk Sınıfları (piksel sayısı):")
            for sinif, piksel in suitability_classes.items():
                yuzde = (piksel / total_score.size) * 100
                print(f"{sinif}: {yuzde:.2f}%")
            
            # Görselleştirme
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            show(slope_score, cmap='RdYlGn', title='Eğim Uygunluğu')
            plt.colorbar(label='Uygunluk Skoru')
            
            plt.subplot(132)
            show(height_score, cmap='RdYlGn', title='Yükseklik Uygunluğu')
            plt.colorbar(label='Uygunluk Skoru')
            
            plt.subplot(133)
            show(total_score, cmap='RdYlGn', title='Toplam Uygunluk')
            plt.colorbar(label='Uygunluk Skoru')
            
            plt.tight_layout()
            plt.savefig("data/analiz/cesme_uygunluk_analizi.png")
            print("Uygunluk analizi görselleştirmesi kaydedildi.")
            
    except Exception as e:
        print(f"Uygunluk analizi sırasında hata: {str(e)}")

def main():
    """Ana fonksiyon"""
    # Analiz klasörünü oluştur
    os.makedirs("data/analiz", exist_ok=True)
    
    # Uygunluk analizini yap
    uygunluk_analizi()
    
    print("\nUygunluk analizi tamamlandı. Sonuçlar data/analiz klasöründe kaydedildi.")

if __name__ == "__main__":
    main() 