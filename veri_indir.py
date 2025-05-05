import os
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

def cesme_sinirlari():
    """Çeşme ilçesinin yaklaşık sınırları"""
    return {
        'north': 38.35,
        'south': 38.25,
        'east': 26.42,
        'west': 26.17
    }

def osm_verileri_indir():
    """OpenStreetMap'ten yol ve diğer verileri indir"""
    print("OpenStreetMap verilerini indirme başladı...")
    
    # Çeşme sınırları
    sinirlar = cesme_sinirlari()
    
    # Yol verilerini indir
    try:
        print("Yol verilerini indirme...")
        G = ox.graph_from_bbox(
            sinirlar['north'], sinirlar['south'],
            sinirlar['east'], sinirlar['west'],
            network_type='drive'
        )
        # GeoDataFrame'e dönüştür
        nodes, edges = ox.graph_to_gdfs(G)
        edges.to_file("data/altyapi/cesme_roads.shp")
        print("Yol verileri indirildi ve kaydedildi.")
    except Exception as e:
        print(f"Yol verilerini indirirken hata: {str(e)}")
    
    # Diğer OSM verilerini indir
    tags = {
        'landuse': True,  # Arazi kullanımı
        'natural': True,  # Doğal alanlar
        'water': True,    # Su kaynakları
        'building': True  # Binalar
    }
    
    try:
        print("Arazi kullanım verilerini indirme...")
        landuse = ox.geometries_from_bbox(
            sinirlar['north'], sinirlar['south'],
            sinirlar['east'], sinirlar['west'],
            tags
        )
        landuse.to_file("data/imar/cesme_landuse.shp")
        print("Arazi kullanım verileri indirildi ve kaydedildi.")
    except Exception as e:
        print(f"Arazi kullanım verilerini indirirken hata: {str(e)}")

def dem_verisi_indir():
    """SRTM DEM verisini indir ve işle"""
    print("DEM verisi indirme başladı...")
    
    try:
        # Çeşme sınırları
        sinirlar = cesme_sinirlari()
        
        # DEM verisi indir
        import subprocess
        cmd = f"eio clip -o data/topografya/cesme_dem.tif --bounds {sinirlar['west']} {sinirlar['south']} {sinirlar['east']} {sinirlar['north']}"
        subprocess.run(cmd, shell=True)
        
        # Eğim hesapla
        if os.path.exists("data/topografya/cesme_dem.tif"):
            with rasterio.open("data/topografya/cesme_dem.tif") as src:
                elevation = src.read(1)
                
                # Eğim hesapla (derece cinsinden)
                dx = np.gradient(elevation, axis=1)
                dy = np.gradient(elevation, axis=0)
                slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
                
                # Eğim verisini kaydet
                with rasterio.open(
                    "data/topografya/cesme_slope.tif",
                    'w',
                    driver='GTiff',
                    height=slope.shape[0],
                    width=slope.shape[1],
                    count=1,
                    dtype=slope.dtype,
                    crs=src.crs,
                    transform=src.transform,
                ) as dst:
                    dst.write(slope, 1)
        
        print("DEM ve eğim verileri indirildi ve işlendi.")
    except Exception as e:
        print(f"DEM verisi indirirken hata: {str(e)}")

def main():
    """Ana fonksiyon"""
    # Klasör yapısını oluştur
    for klasor in ['kadastro', 'imar', 'altyapi', 'topografya', 'jeoloji', 'koruma', 'egitim']:
        os.makedirs(f"data/{klasor}", exist_ok=True)
    
    # Verileri indir
    osm_verileri_indir()
    dem_verisi_indir()
    
    print("Veri indirme işlemi tamamlandı.")

if __name__ == "__main__":
    main() 