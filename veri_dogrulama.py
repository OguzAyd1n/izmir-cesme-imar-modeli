import os
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from datetime import datetime

def validate_and_transform_data(data_path='data'):
    """
    Veri şemasına göre verileri doğrular ve dönüştürür
    """
    # Şema dosyasını oku
    with open(os.path.join(data_path, 'veri_sema.json'), 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    for kategori, dosyalar in schema.items():
        kategori_path = os.path.join(data_path, kategori)
        if not os.path.exists(kategori_path):
            print(f"Uyarı: {kategori} klasörü bulunamadı!")
            continue
            
        for dosya, sema in dosyalar.items():
            dosya_path = os.path.join(kategori_path, dosya)
            if not os.path.exists(dosya_path):
                print(f"Uyarı: {dosya_path} bulunamadı!")
                continue
                
            try:
                # Shapefile'ı oku
                gdf = gpd.read_file(dosya_path)
                
                # CRS kontrolü ve dönüşümü
                if gdf.crs is None:
                    print(f"Uyarı: {dosya} CRS tanımlı değil. EPSG:32635 atanıyor...")
                    gdf.set_crs(epsg=32635, inplace=True)
                elif gdf.crs.to_epsg() != 32635:
                    print(f"Uyarı: {dosya} CRS dönüşümü yapılıyor...")
                    gdf = gdf.to_crs(epsg=32635)
                
                # Alan hesapla ve kontrol et
                if 'ALAN' in sema['fields']:
                    gdf['ALAN'] = gdf.geometry.area
                    print(f"{dosya}: Alan hesaplandı")
                
                # Tarih alanlarını dönüştür
                for field, field_type in sema['fields'].items():
                    if field_type == 'date' and field in gdf.columns:
                        gdf[field] = pd.to_datetime(gdf[field])
                        print(f"{dosya}: {field} tarih formatına dönüştürüldü")
                
                # Veri tiplerini kontrol et ve dönüştür
                for field, field_type in sema['fields'].items():
                    if field not in gdf.columns:
                        print(f"Uyarı: {dosya} içinde {field} alanı bulunamadı!")
                        continue
                        
                    try:
                        if field_type == 'int':
                            gdf[field] = gdf[field].fillna(0).astype(int)
                        elif field_type == 'float':
                            gdf[field] = gdf[field].fillna(0.0).astype(float)
                        elif field_type == 'string':
                            gdf[field] = gdf[field].fillna('').astype(str)
                    except Exception as e:
                        print(f"Hata: {dosya} içinde {field} alanı dönüştürülemedi: {str(e)}")
                
                # Geometri geçerliliğini kontrol et
                invalid_geoms = gdf[~gdf.geometry.is_valid]
                if len(invalid_geoms) > 0:
                    print(f"Uyarı: {dosya} içinde {len(invalid_geoms)} geçersiz geometri bulundu!")
                    # Geometrileri düzelt
                    gdf.geometry = gdf.geometry.buffer(0)
                
                # Dosyayı kaydet
                output_path = os.path.join(data_path, f"processed_{dosya}")
                gdf.to_file(output_path)
                print(f"{dosya} işlendi ve kaydedildi: {output_path}")
                
            except Exception as e:
                print(f"Hata: {dosya} işlenirken bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    validate_and_transform_data() 