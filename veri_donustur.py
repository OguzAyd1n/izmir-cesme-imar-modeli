import geopandas as gpd
import pandas as pd
import os

def yol_verilerini_donustur():
    """Yol verilerini şema ile uyumlu hale getir"""
    print("Yol verileri dönüştürülüyor...")
    
    try:
        # Mevcut veriyi oku
        roads = gpd.read_file("data/altyapi/cesme_roads.shp")
        
        # Şema ile uyumlu yeni veri çerçevesi oluştur
        yeni_roads = pd.DataFrame()
        yeni_roads['OBJECTID'] = range(1, len(roads) + 1)
        yeni_roads['YOL_SINIFI'] = roads['highway']
        yeni_roads['GENISLIK'] = 0.0  # Varsayılan değer
        yeni_roads['KAPLAMA'] = roads['surface']
        yeni_roads['geometry'] = roads['geometry']
        
        # GeoDataFrame'e dönüştür
        yeni_roads = gpd.GeoDataFrame(yeni_roads, geometry='geometry')
        yeni_roads.crs = roads.crs
        
        # Kaydet
        yeni_roads.to_file("data/altyapi/cesme_roads.shp")
        print("Yol verileri dönüştürüldü.")
        
    except Exception as e:
        print(f"Yol verileri dönüştürülürken hata: {str(e)}")

def arazi_kullanim_verilerini_donustur():
    """Arazi kullanım verilerini şema ile uyumlu hale getir"""
    print("Arazi kullanım verileri dönüştürülüyor...")
    
    try:
        # Mevcut veriyi oku
        landuse = gpd.read_file("data/imar/cesme_landuse.shp")
        
        # Şema ile uyumlu yeni veri çerçevesi oluştur
        yeni_zoning = pd.DataFrame()
        yeni_zoning['OBJECTID'] = range(1, len(landuse) + 1)
        yeni_zoning['PLAN_ADI'] = 'Çeşme İlçesi'
        yeni_zoning['PLAN_ONAY'] = pd.Timestamp.now()
        yeni_zoning['FONKSIYON'] = landuse['type']
        yeni_zoning['KAKS'] = 0.0  # Varsayılan değer
        yeni_zoning['TAKS'] = 0.0  # Varsayılan değer
        yeni_zoning['YUKSEKLIK'] = 0.0  # Varsayılan değer
        yeni_zoning['MIN_PARSEL'] = 0.0  # Varsayılan değer
        yeni_zoning['geometry'] = landuse['geometry']
        
        # GeoDataFrame'e dönüştür
        yeni_zoning = gpd.GeoDataFrame(yeni_zoning, geometry='geometry')
        yeni_zoning.crs = landuse.crs
        
        # Kaydet
        yeni_zoning.to_file("data/imar/cesme_zoning.shp")
        print("Arazi kullanım verileri dönüştürüldü.")
        
    except Exception as e:
        print(f"Arazi kullanım verileri dönüştürülürken hata: {str(e)}")

def main():
    """Ana fonksiyon"""
    # Verileri dönüştür
    yol_verilerini_donustur()
    arazi_kullanim_verilerini_donustur()
    
    print("Veri dönüştürme işlemi tamamlandı.")

if __name__ == "__main__":
    main() 