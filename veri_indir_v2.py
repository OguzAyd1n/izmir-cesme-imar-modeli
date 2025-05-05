import os
import requests
import zipfile
import io
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, Polygon, LineString
import json
import shutil

def cesme_sinirlari():
    """Çeşme ilçesinin yaklaşık sınırları"""
    return {
        'north': 38.35,
        'south': 38.25,
        'east': 26.42,
        'west': 26.17
    }

def overpass_verisi_indir():
    """OpenStreetMap verilerini Overpass API üzerinden indir"""
    print("OpenStreetMap verilerini indirme başladı...")
    
    sinirlar = cesme_sinirlari()
    
    # Overpass sorgusu
    query = f"""
    [out:json][timeout:300];
    (
        way["highway"]({sinirlar['south']},{sinirlar['west']},{sinirlar['north']},{sinirlar['east']});
        way["landuse"]({sinirlar['south']},{sinirlar['west']},{sinirlar['north']},{sinirlar['east']});
        way["natural"]({sinirlar['south']},{sinirlar['west']},{sinirlar['north']},{sinirlar['east']});
        way["water"]({sinirlar['south']},{sinirlar['west']},{sinirlar['north']},{sinirlar['east']});
        way["building"]({sinirlar['south']},{sinirlar['west']},{sinirlar['north']},{sinirlar['east']});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        print("Veri indiriliyor...")
        response = requests.get(
            'http://overpass-api.de/api/interpreter',
            params={'data': query}
        )
        data = response.json()
        
        # Yolları ayır
        roads = []
        landuse = []
        
        for element in data['elements']:
            if element['type'] == 'way':
                coords = []
                if 'nodes' in element:
                    for node_id in element['nodes']:
                        node = next((n for n in data['elements'] if n['type'] == 'node' and n['id'] == node_id), None)
                        if node:
                            coords.append([node['lon'], node['lat']])
                            
                if len(coords) > 1:
                    if 'tags' in element and 'highway' in element['tags']:
                        # Yollar için LineString kullan
                        roads.append({
                            'geometry': LineString(coords),
                            'highway': element['tags'].get('highway', 'unknown'),
                            'name': element['tags'].get('name', 'unnamed'),
                            'surface': element['tags'].get('surface', 'unknown')
                        })
                    elif 'tags' in element and ('landuse' in element['tags'] or 'natural' in element['tags'] or 'water' in element['tags'] or 'building' in element['tags']):
                        if coords[0] == coords[-1]:  # Kapalı poligon ise
                            landuse.append({
                                'geometry': Polygon(coords),
                                'type': element['tags'].get('landuse', element['tags'].get('natural', element['tags'].get('water', element['tags'].get('building', 'unknown')))),
                                'name': element['tags'].get('name', 'unnamed')
                            })
        
        # GeoDataFrame'lere dönüştür
        if roads:
            roads_gdf = gpd.GeoDataFrame(roads, geometry='geometry')
            roads_gdf.set_crs(epsg=4326, inplace=True)
            roads_gdf.to_crs(epsg=32635, inplace=True)
            roads_gdf.to_file("data/altyapi/cesme_roads.shp")
            print("Yol verileri kaydedildi.")
            
        if landuse:
            landuse_gdf = gpd.GeoDataFrame(landuse, geometry='geometry')
            landuse_gdf.set_crs(epsg=4326, inplace=True)
            landuse_gdf.to_crs(epsg=32635, inplace=True)
            landuse_gdf.to_file("data/imar/cesme_landuse.shp")
            print("Arazi kullanım verileri kaydedildi.")
            
    except Exception as e:
        print(f"Veri indirirken hata: {str(e)}")

def srtm_verisi_indir():
    """SRTM verilerini NASA sunucusundan indir"""
    print("SRTM verisi indirme başladı...")
    
    # Çeşme için gerekli SRTM karesi (N38E026)
    srtm_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_41_04.zip"
    
    try:
        # Hedef dosya yolu
        hedef_dosya = "data/topografya/cesme_dem.tif"
        
        # Eğer hedef dosya varsa sil
        if os.path.exists(hedef_dosya):
            os.remove(hedef_dosya)
        
        # Temp klasörünü temizle ve oluştur
        temp_dir = "data/topografya/temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        print("DEM verisi indiriliyor...")
        response = requests.get(srtm_url)
        
        if response.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(temp_dir)
            
            # DEM dosyasını bul ve işle
            for file in os.listdir(temp_dir):
                if file.endswith(".tif"):
                    # Dosyayı taşı ve yeniden adlandır
                    shutil.move(
                        os.path.join(temp_dir, file),
                        hedef_dosya
                    )
                    print("DEM verisi indirildi ve kaydedildi.")
                    break
            
            # Temp klasörünü temizle
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"SRTM verisi indirirken hata: {str(e)}")
        # Hata durumunda temp klasörünü temizle
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    """Ana fonksiyon"""
    # Klasör yapısını oluştur
    for klasor in ['kadastro', 'imar', 'altyapi', 'topografya', 'jeoloji', 'koruma', 'egitim']:
        os.makedirs(f"data/{klasor}", exist_ok=True)
    
    # Verileri indir
    overpass_verisi_indir()
    srtm_verisi_indir()
    
    print("Veri indirme işlemi tamamlandı.")

if __name__ == "__main__":
    main() 