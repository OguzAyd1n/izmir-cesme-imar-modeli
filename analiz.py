import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
import os

def topografik_analiz():
    """Topografya verilerini analiz et"""
    print("Topografik analiz başladı...")
    
    try:
        # DEM verisini oku
        with rasterio.open("data/topografya/cesme_dem.tif") as src:
            dem = src.read(1)
            
            # Temel istatistikler
            min_yukseklik = np.min(dem)
            max_yukseklik = np.max(dem)
            ortalama_yukseklik = np.mean(dem)
            
            print(f"Yükseklik istatistikleri:")
            print(f"Minimum yükseklik: {min_yukseklik:.2f} m")
            print(f"Maksimum yükseklik: {max_yukseklik:.2f} m")
            print(f"Ortalama yükseklik: {ortalama_yukseklik:.2f} m")
            
            # Eğim hesapla
            dx = np.gradient(dem, axis=1)
            dy = np.gradient(dem, axis=0)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            
            # Eğim sınıfları
            slope_classes = {
                'Düz (0-5°)': np.sum((slope >= 0) & (slope < 5)),
                'Hafif Eğimli (5-15°)': np.sum((slope >= 5) & (slope < 15)),
                'Orta Eğimli (15-25°)': np.sum((slope >= 15) & (slope < 25)),
                'Dik (25-45°)': np.sum((slope >= 25) & (slope < 45)),
                'Çok Dik (>45°)': np.sum(slope >= 45)
            }
            
            print("\nEğim sınıfları:")
            for sinif, piksel in slope_classes.items():
                yuzde = (piksel / slope.size) * 100
                print(f"{sinif}: {yuzde:.2f}%")
            
            # Görselleştirme
            plt.figure(figsize=(15, 5))
            
            plt.subplot(121)
            show(dem, cmap='terrain', title='Yükseklik (m)')
            plt.colorbar(label='Yükseklik (m)')
            
            plt.subplot(122)
            show(slope, cmap='viridis', title='Eğim (derece)')
            plt.colorbar(label='Eğim (derece)')
            
            plt.tight_layout()
            plt.savefig("data/topografya/cesme_topografya_analiz.png")
            print("Topografik analiz görselleştirmesi kaydedildi.")
            
    except Exception as e:
        print(f"Topografik analiz sırasında hata: {str(e)}")

def arazi_kullanim_analiz():
    """Arazi kullanım verilerini analiz et"""
    print("Arazi kullanım analizi başladı...")
    
    try:
        # Arazi kullanım verilerini oku
        zoning = gpd.read_file("data/imar/cesme_zoning.shp")
        
        # Fonksiyon sınıflarına göre alan hesapla
        fonksiyon_alanlari = zoning.groupby('FONKSIYON')['geometry'].apply(
            lambda x: x.area.sum() / 10000  # m²'den hektara çevir
        )
        
        print("\nArazi Kullanım Alanları (hektar):")
        for fonksiyon, alan in fonksiyon_alanlari.items():
            print(f"{fonksiyon}: {alan:.2f} ha")
        
        # Görselleştirme
        plt.figure(figsize=(10, 10))
        zoning.plot(column='FONKSIYON', legend=True, cmap='tab20')
        plt.title('Arazi Kullanım Sınıfları')
        plt.axis('off')
        plt.savefig("data/imar/cesme_arazi_kullanim.png")
        print("Arazi kullanım görselleştirmesi kaydedildi.")
        
    except Exception as e:
        print(f"Arazi kullanım analizi sırasında hata: {str(e)}")

def yol_analiz():
    """Yol verilerini analiz et"""
    print("Yol analizi başladı...")
    
    try:
        # Yol verilerini oku
        roads = gpd.read_file("data/altyapi/cesme_roads.shp")
        
        # Yol sınıflarına göre uzunluk hesapla
        yol_uzunluklari = roads.groupby('YOL_SINIFI')['geometry'].apply(
            lambda x: x.length.sum() / 1000  # metre'den kilometreye çevir
        )
        
        print("\nYol Uzunlukları (km):")
        for sinif, uzunluk in yol_uzunluklari.items():
            print(f"{sinif}: {uzunluk:.2f} km")
        
        # Görselleştirme
        plt.figure(figsize=(10, 10))
        roads.plot(column='YOL_SINIFI', legend=True, cmap='tab20')
        plt.title('Yol Ağı')
        plt.axis('off')
        plt.savefig("data/altyapi/cesme_yol_agi.png")
        print("Yol ağı görselleştirmesi kaydedildi.")
        
    except Exception as e:
        print(f"Yol analizi sırasında hata: {str(e)}")

def main():
    """Ana fonksiyon"""
    # Analizleri yap
    topografik_analiz()
    arazi_kullanim_analiz()
    yol_analiz()
    
    print("\nAnaliz işlemi tamamlandı. Sonuçlar ilgili klasörlerde kaydedildi.")

if __name__ == "__main__":
    main() 