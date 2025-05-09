# İzmir/Çeşme İmara Açılabilecek Arsa Tespit Modeli
# İlk 5 Adım Uygulaması

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, shape
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
# from osgeo import gdal, ogr, osr
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')
import json

# ------------ ADIM 1: VERİ TOPLAMA VE YÜKLEME ------------

def veri_yukle(data_path='data'):
    """
    Model için gerekli tüm verileri yükler ve birleştirir
    """
    print("Veri yükleme işlemi başlatılıyor...")
    
    veri_sozlugu = {}
    
    try:
        # Kadastro verilerini yükle
        parsel_dosyasi = os.path.join(data_path, 'kadastro/cesme_parcels.shp')
        if os.path.exists(parsel_dosyasi):
            parcels = gpd.read_file(parsel_dosyasi)
            # CRS kontrolü ve dönüşümü
            if parcels.crs is None:
                parcels.set_crs(epsg=32635, inplace=True)  # UTM Zone 35N
            elif parcels.crs.to_epsg() != 32635:
                parcels = parcels.to_crs(epsg=32635)
            veri_sozlugu['parcels'] = parcels
        else:
            print(f"Uyarı: {parsel_dosyasi} bulunamadı!")
            
        # İmar planı verilerini yükle
        imar_dosyasi = os.path.join(data_path, 'imar/cesme_zoning.shp')
        if os.path.exists(imar_dosyasi):
            zoning = gpd.read_file(imar_dosyasi)
            if zoning.crs is None:
                zoning.set_crs(epsg=32635, inplace=True)
            elif zoning.crs.to_epsg() != 32635:
                zoning = zoning.to_crs(epsg=32635)
            veri_sozlugu['zoning'] = zoning
        else:
            print(f"Uyarı: {imar_dosyasi} bulunamadı!")
            
        # Altyapı verilerini yükle
        for altyapi_dosyasi in ['roads', 'utilities']:
            dosya_yolu = os.path.join(data_path, f'altyapi/cesme_{altyapi_dosyasi}.shp')
            if os.path.exists(dosya_yolu):
                veri = gpd.read_file(dosya_yolu)
                if veri.crs is None:
                    veri.set_crs(epsg=32635, inplace=True)
                elif veri.crs.to_epsg() != 32635:
                    veri = veri.to_crs(epsg=32635)
                veri_sozlugu[altyapi_dosyasi] = veri
            else:
                print(f"Uyarı: {dosya_yolu} bulunamadı!")
                
        # Topografya verilerini yükle
        for topo_dosyasi in ['dem', 'slope']:
            dosya_yolu = os.path.join(data_path, f'topografya/cesme_{topo_dosyasi}.tif')
            if os.path.exists(dosya_yolu):
                with rasterio.open(dosya_yolu) as src:
                    # CRS kontrolü
                    if src.crs is None or src.crs.to_epsg() != 32635:
                        print(f"Uyarı: {topo_dosyasi} CRS dönüşümü gerekiyor!")
                    veri_sozlugu[topo_dosyasi] = src
            else:
                print(f"Uyarı: {dosya_yolu} bulunamadı!")
                
        # Jeoloji verilerini yükle
        jeoloji_dosyasi = os.path.join(data_path, 'jeoloji/cesme_geology.shp')
        if os.path.exists(jeoloji_dosyasi):
            geology = gpd.read_file(jeoloji_dosyasi)
            if geology.crs is None:
                geology.set_crs(epsg=32635, inplace=True)
            elif geology.crs.to_epsg() != 32635:
                geology = geology.to_crs(epsg=32635)
            veri_sozlugu['geology'] = geology
        else:
            print(f"Uyarı: {jeoloji_dosyasi} bulunamadı!")
            
        # Koruma alanları verilerini yükle
        koruma_dosyasi = os.path.join(data_path, 'koruma/cesme_protected.shp')
        if os.path.exists(koruma_dosyasi):
            protected = gpd.read_file(koruma_dosyasi)
            if protected.crs is None:
                protected.set_crs(epsg=32635, inplace=True)
            elif protected.crs.to_epsg() != 32635:
                protected = protected.to_crs(epsg=32635)
            veri_sozlugu['protected'] = protected
        else:
            print(f"Uyarı: {koruma_dosyasi} bulunamadı!")
            
        # Eğitim verilerini yükle
        egitim_dosyasi = os.path.join(data_path, 'egitim/past_zoning_changes.shp')
        if os.path.exists(egitim_dosyasi):
            training_data = gpd.read_file(egitim_dosyasi)
            if training_data.crs is None:
                training_data.set_crs(epsg=32635, inplace=True)
            elif training_data.crs.to_epsg() != 32635:
                training_data = training_data.to_crs(epsg=32635)
            veri_sozlugu['training_data'] = training_data
        else:
            print(f"Uyarı: {egitim_dosyasi} bulunamadı!")
            
        # Veri doğrulama
        gerekli_veriler = ['parcels', 'zoning']
        eksik_veriler = [v for v in gerekli_veriler if v not in veri_sozlugu]
        
        if eksik_veriler:
            print(f"Kritik Hata: Şu gerekli veriler eksik: {', '.join(eksik_veriler)}")
            print("Örnek veriler kullanılacak...")
            return dummy_veri_olustur()
            
        print("Veri yükleme işlemi başarıyla tamamlandı.")
        return veri_sozlugu
        
    except Exception as e:
        print(f"Veri yükleme hatası: {str(e)}")
        print("Örnek veriler kullanılacak...")
        return dummy_veri_olustur()

def dummy_veri_olustur():
    """
    Test amaçlı örnek veri oluşturur (gerçek veri olmadığında)
    """
    print("Örnek test verileri oluşturuluyor...")
    
    # Çeşme'nin yaklaşık koordinat aralıkları
    minx, miny = 469000, 4232000  # Yaklaşık UTM koordinatları (Zone 35N)
    maxx, maxy = 482000, 4244000
    
    # Rastgele parsel oluştur
    np.random.seed(42)
    n_parcels = 1000
    
    # Parsel geometrileri oluştur
    geometries = []
    for i in range(n_parcels):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        width = np.random.uniform(20, 100)
        height = np.random.uniform(20, 100)
        
        polygon = Polygon([
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ])
        geometries.append(polygon)
    
    # Parsel verileri oluştur
    parcels_data = {
        'geometry': geometries,
        'parcel_id': [f'P{i:05d}' for i in range(n_parcels)],
        'area': [poly.area for poly in geometries],
        'perimeter': [poly.length for poly in geometries],
        'owner_type': np.random.choice(['private', 'public', 'treasury'], n_parcels),
        'current_use': np.random.choice(['empty', 'agricultural', 'residential', 'commercial'], n_parcels)
    }
    
    parcels = gpd.GeoDataFrame(parcels_data, crs="EPSG:32635")
    
    # İmar verileri oluştur
    zoning_data = {
        'geometry': parcels.geometry,
        'zone_type': np.random.choice(['residential', 'commercial', 'industrial', 'agricultural', 'tourism', 'public'], n_parcels),
        'KAKS': np.random.uniform(0.5, 2.0, n_parcels),
        'TAKS': np.random.uniform(0.1, 0.4, n_parcels),
        'height_limit': np.random.choice([6.5, 9.5, 12.5, 15.5], n_parcels),
        'min_parcel': np.random.choice([300, 500, 1000, 2000], n_parcels),
        'is_currently_zoned': np.random.choice([True, False], n_parcels, p=[0.7, 0.3]),
    }
    
    zoning = gpd.GeoDataFrame(zoning_data, crs="EPSG:32635")
    
    # Eğitim verileri oluştur (geçmişte imara açılan parseller)
    n_training = 300
    training_indices = np.random.choice(n_parcels, n_training, replace=False)
    
    training_data = {
        'geometry': [parcels.geometry[i] for i in training_indices],
        'parcel_id': [parcels.parcel_id[i] for i in training_indices],
        'year_zoned': np.random.randint(2000, 2024, n_training),
        'previous_use': np.random.choice(['agricultural', 'empty', 'forest'], n_training),
        'new_use': np.random.choice(['residential', 'tourism', 'commercial'], n_training),
        'was_zoned': np.random.choice([1, 0], n_training, p=[0.8, 0.2])
    }
    
    training_df = gpd.GeoDataFrame(training_data, crs="EPSG:32635")
    
    # Koruma alanları oluştur
    n_protected = 10
    protected_polys = []
    protected_types = []
    
    for i in range(n_protected):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        radius = np.random.uniform(200, 1000)
        
        # Daire şeklinde koruma alanları oluştur
        circle_points = []
        for angle in range(0, 360, 10):
            rad = np.radians(angle)
            px = x + radius * np.cos(rad)
            py = y + radius * np.sin(rad)
            circle_points.append((px, py))
        
        protected_polys.append(Polygon(circle_points))
        protected_types.append(np.random.choice(['natural', 'archaeological', 'urban', 'coastal', 'forest']))
    
    protected_data = {
        'geometry': protected_polys,
        'protection_type': protected_types,
        'protection_level': np.random.choice(['1', '2', '3'], n_protected),
        'year_designated': np.random.randint(1980, 2024, n_protected)
    }
    
    protected = gpd.GeoDataFrame(protected_data, crs="EPSG:32635")
    
    # Jeoloji verileri
    n_geology = 15
    geology_polys = []
    geology_types = []
    
    for i in range(n_geology):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        width = np.random.uniform(500, 2000)
        height = np.random.uniform(500, 2000)
        
        polygon = Polygon([
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ])
        geology_polys.append(polygon)
        geology_types.append(np.random.choice(['alluvial', 'limestone', 'andesite', 'schist', 'granite']))
    
    geology_data = {
        'geometry': geology_polys,
        'rock_type': geology_types,
        'soil_class': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_geology),
        'earthquake_risk': np.random.choice(['low', 'medium', 'high'], n_geology)
    }
    
    geology = gpd.GeoDataFrame(geology_data, crs="EPSG:32635")
    
    # Yol verileri
    n_roads = 50
    road_lines = []
    road_types = []
    
    for i in range(n_roads):
        x1 = np.random.uniform(minx, maxx)
        y1 = np.random.uniform(miny, maxy)
        length = np.random.uniform(100, 3000)
        angle = np.random.uniform(0, 360)
        
        rad = np.radians(angle)
        x2 = x1 + length * np.cos(rad)
        y2 = y1 + length * np.sin(rad)
        
        road_line = LineString([(x1, y1), (x2, y2)])
        road_lines.append(road_line)
        road_types.append(np.random.choice(['highway', 'main', 'secondary', 'local', 'dirt']))
    
    road_data = {
        'geometry': road_lines,
        'road_type': road_types,
        'width': np.random.uniform(3, 30, n_roads),
        'surface': np.random.choice(['asphalt', 'concrete', 'gravel', 'dirt'], n_roads)
    }
    
    roads = gpd.GeoDataFrame(road_data, crs="EPSG:32635")
    
    # Altyapı verileri (noktalar)
    n_utilities = 100
    utility_points = []
    utility_types = []
    
    for i in range(n_utilities):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        utility_points.append(Point(x, y))
        utility_types.append(np.random.choice(['water', 'electricity', 'sewage', 'internet']))
    
    utility_data = {
        'geometry': utility_points,
        'utility_type': utility_types,
        'capacity': np.random.uniform(10, 100, n_utilities),
        'year_built': np.random.randint(1970, 2024, n_utilities)
    }
    
    utilities = gpd.GeoDataFrame(utility_data, crs="EPSG:32635")
    
    # DEM ve eğim verilerini simüle et (gerçek uygulamada rasterio nesneleri olacak)
    class DummyRaster:
        def __init__(self, bounds):
            self.bounds = bounds
            self.height = 1000
            self.width = 1000
            self.transform = rasterio.transform.from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3], 1000, 1000)
        
        def read(self, band=1):
            # Rastgele yükseklik ve eğim verileri oluştur
            return [np.random.uniform(0, 500, (1000, 1000))]
    
    dem = DummyRaster((minx, miny, maxx, maxy))
    slope = DummyRaster((minx, miny, maxx, maxy))
    
    print("Örnek test verileri oluşturuldu.")
    
    return {
        'parcels': parcels,
        'zoning': zoning,
        'roads': roads,
        'utilities': utilities,
        'dem': dem,
        'slope': slope,
        'geology': geology,
        'protected': protected,
        'training_data': training_df
    }

# ------------ ADIM 2: VERİ ÖN İŞLEME ------------

def veri_on_isleme(veri_sozlugu):
    """
    Verileri temizler, dönüştürür ve CBS veritabanına entegre eder
    """
    print("Veri ön işleme başlatılıyor...")
    
    # CRS kontrolü ve dönüşümü
    target_crs = "EPSG:32635"  # UTM Zone 35N (İzmir/Çeşme için uygun)
    
    for key, gdf in veri_sozlugu.items():
        if key not in ['dem', 'slope']:  # Raster veriler hariç
            if hasattr(gdf, 'crs'):
                if gdf.crs is None:
                    gdf.crs = target_crs
                elif gdf.crs != target_crs:
                    print(f"{key} verisi için CRS dönüşümü yapılıyor: {gdf.crs} -> {target_crs}")
                    gdf = gdf.to_crs(target_crs)
                    veri_sozlugu[key] = gdf
    
    # Parsel verileri üzerinde işlemler
    parcels = veri_sozlugu['parcels']
    
    # Eksik verileri doldur
    if 'area' not in parcels.columns:
        parcels['area'] = parcels.geometry.area
    
    if 'perimeter' not in parcels.columns:
        parcels['perimeter'] = parcels.geometry.length
    
    # Şekil indeksini hesapla (ne kadar 1'e yakınsa o kadar kare/daire şeklinde)
    parcels['shape_index'] = (4 * np.pi * parcels['area']) / (parcels['perimeter'] ** 2)
    
    # İmar verileri ile parselleri birleştir
    zoning = veri_sozlugu['zoning']
    parcels = gpd.sjoin(parcels, zoning, how='left', predicate='intersects')
    
    # Korunan alanlarla kesişimi kontrol et
    protected = veri_sozlugu['protected']

    # Çakışmayı önle
    for df in [parcels, protected]:
      if 'index_right' in df.columns:
        df.drop(columns='index_right', inplace=True)

    # Sonra spatial join
    protected_intersection = gpd.sjoin(parcels, protected, how='left', predicate='intersects')
    protected_indices = protected_intersection.index.unique()
    parcels['is_protected'] = parcels.index.isin(protected_indices)
    protection_type_map = protected_intersection[['protection_type']].groupby(protected_intersection.index).first()
    parcels['protection_type'] = parcels.index.map(protection_type_map['protection_type'])

    # Jeolojik verilerle birleştir
    geology = veri_sozlugu['geology']
    parcels = gpd.sjoin(parcels, geology, how='left', predicate='intersects')
    
    # Yollara olan mesafeyi hesapla
    roads = veri_sozlugu['roads']
    
    def min_distance_to_roads(geom, roads_gdf):
        min_dist = float('inf')
        for road_geom in roads_gdf.geometry:
            dist = geom.distance(road_geom)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    parcels['distance_to_road'] = parcels.geometry.apply(lambda x: min_distance_to_roads(x, roads))
    
    # Altyapı tesislerine olan mesafeyi hesapla
    utilities = veri_sozlugu['utilities']
    
    def min_distance_to_utilities(geom, utilities_gdf, utility_type=None):
        if utility_type:
            utilities_filtered = utilities_gdf[utilities_gdf['utility_type'] == utility_type]
        else:
            utilities_filtered = utilities_gdf
            
        min_dist = float('inf')
        for util_geom in utilities_filtered.geometry:
            dist = geom.distance(util_geom)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    parcels['distance_to_water'] = parcels.geometry.apply(
        lambda x: min_distance_to_utilities(x, utilities, 'water'))
    parcels['distance_to_electricity'] = parcels.geometry.apply(
        lambda x: min_distance_to_utilities(x, utilities, 'electricity'))
    parcels['distance_to_sewage'] = parcels.geometry.apply(
        lambda x: min_distance_to_utilities(x, utilities, 'sewage'))
    
    # DEM ve eğim verilerini parsel bazında çıkar
    dem_raster = veri_sozlugu['dem']
    slope_raster = veri_sozlugu['slope']
    
    def extract_raster_stats(geom, raster):
        try:
            # Eğer gerçek raster nesnesi ise
            if hasattr(raster, 'read') and callable(getattr(raster, 'read')):
                # Geometriyi raster CRS'ye dönüştür
                if hasattr(raster, 'crs') and raster.crs:
                    # Burada raster.crs bir CRS nesnesi olmalı
                    geom_json = [mapping(geom)]
                    out_image, out_transform = mask(raster, geom_json, crop=True)
                    # Mask değerlerini al
                    out_image = np.ma.masked_array(out_image, mask=(out_image==raster.nodata))
                    # İstatistikleri hesapla
                    return {
                        'min': float(np.min(out_image)),
                        'max': float(np.max(out_image)),
                        'mean': float(np.mean(out_image)),
                        'median': float(np.median(out_image)),
                        'std': float(np.std(out_image))
                    }
                else:
                    # Gerçek raster olmadığında rastgele değerler dön
                    return {
                        'min': float(np.random.uniform(0, 100)),
                        'max': float(np.random.uniform(100, 500)),
                        'mean': float(np.random.uniform(50, 200)),
                        'median': float(np.random.uniform(50, 200)),
                        'std': float(np.random.uniform(5, 50))
                    }
            else:
                # Eğer dummy raster ise rastgele değerler üret
                return {
                    'min': float(np.random.uniform(0, 100)),
                    'max': float(np.random.uniform(100, 500)),
                    'mean': float(np.random.uniform(50, 200)),
                    'median': float(np.random.uniform(50, 200)),
                    'std': float(np.random.uniform(5, 50))
                }
        except Exception as e:
            print(f"Raster işleme hatası: {e}")
            # Hata durumunda varsayılan değerler
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0
            }
    
    # Her parsel için yükseklik ve eğim istatistiklerini hesapla
    elevation_stats = parcels.geometry.apply(lambda x: extract_raster_stats(x, dem_raster))
    slope_stats = parcels.geometry.apply(lambda x: extract_raster_stats(x, slope_raster))
    
    parcels['elevation_min'] = elevation_stats.apply(lambda x: x['min'])
    parcels['elevation_max'] = elevation_stats.apply(lambda x: x['max'])
    parcels['elevation_mean'] = elevation_stats.apply(lambda x: x['mean'])
    parcels['elevation_std'] = elevation_stats.apply(lambda x: x['std'])
    
    parcels['slope_min'] = slope_stats.apply(lambda x: x['min'])
    parcels['slope_max'] = slope_stats.apply(lambda x: x['max'])
    parcels['slope_mean'] = slope_stats.apply(lambda x: x['mean'])
    parcels['slope_std'] = slope_stats.apply(lambda x: x['std'])
    
    # Eksik değerleri doldur
    parcels = parcels.fillna({
        'is_protected': False,
        'distance_to_road': parcels['distance_to_road'].max(),
        'distance_to_water': parcels['distance_to_water'].max(),
        'distance_to_electricity': parcels['distance_to_electricity'].max(),
        'distance_to_sewage': parcels['distance_to_sewage'].max(),
        'elevation_mean': parcels['elevation_mean'].median(),
        'slope_mean': parcels['slope_mean'].median(),
    })
    
    # Kategorik değişkenleri doldur
    for col in ['owner_type', 'current_use', 'zone_type', 'protection_type', 'rock_type', 'soil_class']:
        if col in parcels.columns and parcels[col].isna().any():
            parcels[col] = parcels[col].fillna('unknown')
    
    # Veri tipi dönüşümleri
    for col in parcels.columns:
        if col not in ['geometry']:
            if 'int' in str(parcels[col].dtype):
                parcels[col] = parcels[col].astype(float)
    
    # Gereksiz sütunları temizle
    if 'index_right' in parcels.columns:
        parcels = parcels.drop(columns=['index_right'])
    
    print("Veri ön işleme tamamlandı.")
    
    return parcels

# ------------ ADIM 3: ÖZELLİK MÜHENDİSLİĞİ ------------

def ozellik_muhendisligi(parcels_gdf):
    """
    İmara uygunluğu belirleyen özellikler oluşturur
    """
    print("Özellik mühendisliği başlatılıyor...")
    
    # Mevcut özellikleri kopyala
    parcels = parcels_gdf.copy()
    
    # Büyüklük sınıflandırması
    parcels['size_category'] = pd.cut(
        parcels['area'], 
        bins=[0, 300, 500, 1000, 2000, float('inf')],
        labels=['very_small', 'small', 'medium', 'large', 'very_large']
    )
    
    # Eğim kategorisi
    parcels['slope_category'] = pd.cut(
        parcels['slope_mean'],
        bins=[0, 5, 10, 15, 30, float('inf')],
        labels=['flat', 'gentle', 'moderate', 'steep', 'very_steep']
    )
    
    # Yola uzaklık kategorisi
    parcels['road_proximity'] = pd.cut(
        parcels['distance_to_road'],
        bins=[0, 50, 100, 200, 500, float('inf')],
        labels=['immediate', 'close', 'moderate', 'far', 'remote']
    )
    
    # Altyapı erişim puanı
    # Not: Düşük mesafe = yüksek puan (ters orantı)
    max_distance = max(
        parcels['distance_to_water'].max(),
        parcels['distance_to_electricity'].max(),
        parcels['distance_to_sewage'].max()
    )
    
    parcels['water_score'] = 1 - (parcels['distance_to_water'] / max_distance)
    parcels['electricity_score'] = 1 - (parcels['distance_to_electricity'] / max_distance)
    parcels['sewage_score'] = 1 - (parcels['distance_to_sewage'] / max_distance)
    
    # Toplam altyapı puanı
    parcels['infrastructure_score'] = (
        parcels['water_score'] * 0.3 + 
        parcels['electricity_score'] * 0.3 + 
        parcels['sewage_score'] * 0.4
    )
    
    # Şekil uygunluğu (geometri düzgünlüğü - daha yüksek = daha düzgün)
    # Shape index üzerinden hesapla - 1'e ne kadar yakınsa o kadar düzgün
    parcels['shape_suitability'] = 1 - np.minimum(1, np.abs(1 - parcels['shape_index']))
    
    # İmar katsayıları varsa kullan
    if 'KAKS' in parcels.columns and 'TAKS' in parcels.columns:
        parcels['development_potential'] = parcels['KAKS'] * parcels['area']
    else:
        # Yoksa varsayılan değerler ata
        parcels['development_potential'] = parcels['area']
    
    # Jeolojik risk faktörü
    if 'earthquake_risk' in parcels.columns:
        risk_map = {'low': 0.1, 'medium': 0.5, 'high': 0.9, 'unknown': 0.5}
        parcels['geological_risk'] = parcels['earthquake_risk'].map(risk_map)
    else:
        parcels['geological_risk'] = 0.5
    
    # Koruma durumu faktörü (korunan = 0, koruma yok = 1)
    parcels['buildable_factor'] = (~parcels['is_protected']).astype(int)
    
    # Mevcut kullanım uygunluğu
    if 'current_use' in parcels.columns:
        use_suitability = {
            'empty': 1.0,           # Boş arsa - en uygun
            'agricultural': 0.7,    # Tarım arazisi - orta uygun
            'residential': 0.3,     # Mevcut konut - düşük uygun
            'commercial': 0.3,      # Mevcut ticari - düşük uygun
            'unknown': 0.5          # Bilinmeyen - orta
        }
        parcels['use_suitability'] = parcels['current_use'].map(
            lambda x: use_suitability.get(x, 0.5)
        )
    else:
        parcels['use_suitability'] = 0.5
    
    # Eğim uygunluğu (düz arazi = 1, dik eğim = 0)
    parcels['slope_suitability'] = 1 - (parcels['slope_mean'] / 45)  # 45 derece max eğim kabul edildi
    parcels['slope_suitability'] = parcels['slope_suitability'].clip(0, 1)
    
    # İmara açılabilme potansiyeli (ağırlıklandırılmış toplam)
    parcels['zoning_potential'] = (
        parcels['buildable_factor'] * 0.30 +     # Yasal koruma durumu
        parcels['infrastructure_score'] * 0.20 + # Altyapı erişimi
        parcels['slope_suitability'] * 0.15 +    # Eğim uygunluğu
        parcels['shape_suitability'] * 0.10 +    # Parsel şekli
        parcels['use_suitability'] * 0.15 +      # Mevcut kullanım
        (1 - parcels['geological_risk']) * 0.10  # Jeolojik risk (ters orantılı)
    )
    
    print("Özellik mühendisliği tamamlandı.")
    
    return parcels

# ------------ ADIM 4: MODEL GELİŞTİRME ------------

def kural_tabanli_filtreleme(parcels_gdf):
    """
    İmar yönetmeliklerine göre kesin kısıtlama gerektiren alanları filtreler
    """
    print("Kural tabanlı filtreleme başlatılıyor...")
    
    # Filtreleme kriterleri
    parcels = parcels_gdf.copy()
    
    # 1. Korunan alanlardaki parselleri işaretle
    parcels['rule_protected'] = parcels['is_protected']
    
    # 2. Çok dik eğimli arazileri işaretle (genellikle >30 derece yapılaşmaya uygun değil)
    parcels['rule_steep_slope'] = parcels['slope_mean'] > 30
    
    # 3. Minimum parsel büyüklüğü kontrolü
    if 'min_parcel' in parcels.columns:
        parcels['rule_small_area'] = parcels['area'] < parcels['min_parcel']
    else:
        # Varsayılan minimum 300m² olsun
        parcels['rule_small_area'] = parcels['area'] < 300
    
    # 4. Yüksek jeolojik risk alanlarını işaretle
    if 'geological_risk' in parcels.columns:
        parcels['rule_high_geo_risk'] = parcels['geological_risk'] > 0.8
    else:
        parcels['rule_high_geo_risk'] = False
    
    # 5. Şekli çok bozuk olan parselleri işaretle (inşaat için uygun değil)
    parcels['rule_irregular_shape'] = parcels['shape_index'] < 0.4
    
    # 6. Yola çok uzak parselleri işaretle
    parcels['rule_far_from_road'] = parcels['distance_to_road'] > 1000  # 1 km'den uzak
    
    # Tüm kural ihlallerini birleştir
    parcels['has_restrictions'] = (
        parcels['rule_protected'] | 
        parcels['rule_steep_slope'] | 
        parcels['rule_small_area'] | 
        parcels['rule_high_geo_risk'] | 
        parcels['rule_irregular_shape'] |
        parcels['rule_far_from_road']
    )
    
    # Kısıtlama sayısını hesapla
    restriction_cols = [col for col in parcels.columns if col.startswith('rule_')]
    parcels['restriction_count'] = parcels[restriction_cols].sum(axis=1)
    
    print(f"Kural tabanlı filtreleme tamamlandı. {parcels['has_restrictions'].sum()} parsel kısıtlamalı.")
    
    return parcels

def makine_ogrenmesi_modeli(parcels_gdf, training_data=None):
    """
    İmara açılabilirliği tahmin eden makine öğrenmesi modeli
    """
    print("Makine öğrenmesi modeli geliştiriliyor...")
    
    parcels = parcels_gdf.copy()
    
    # Eğitim verisini oluştur
    if training_data is not None and not training_data.empty:
        # Gerçek eğitim verisi varsa kullan
        train_df = training_data.copy()
        if 'was_zoned' in train_df.columns:
            target = 'was_zoned'
        else:
            # Eğitim verisinde hedef değişken yoksa oluştur
            train_df['was_zoned'] = np.random.choice([0, 1], len(train_df), p=[0.3, 0.7])
            target = 'was_zoned'
    else:
        # Eğitim verisi yoksa yapay veri oluştur
        # Yüksek potansiyelli parsellerin imara açılma olasılığı yüksek olsun
        parcels['synthetic_target'] = (parcels['zoning_potential'] > 0.7).astype(int)
        # Rastgelelik ekle
        random_factor = np.random.uniform(0, 0.3, len(parcels))
        parcels['synthetic_target'] = ((parcels['zoning_potential'] + random_factor) > 0.7).astype(int)
        train_df = parcels.copy()
        target = 'synthetic_target'
    
    # Özellik seçimi
    feature_cols = [
        'area', 'shape_index', 'distance_to_road', 
        'distance_to_water', 'distance_to_electricity', 'distance_to_sewage',
        'elevation_mean', 'slope_mean',
        'infrastructure_score', 'shape_suitability', 'slope_suitability',
        'use_suitability', 'geological_risk', 'buildable_factor',
        'restriction_count'
    ]
    
    # Kolonları kontrol et ve eksik olanları çıkar
    available_features = [col for col in feature_cols if col in train_df.columns]
    
    if len(available_features) < 5:
        print("Yeterli özellik yok, temel özellikler oluşturuluyor.")
        # Temel özellikleri oluştur
        if 'area' not in train_df.columns:
            train_df['area'] = train_df.geometry.area
        if 'distance_to_road' not in train_df.columns:
            train_df['distance_to_road'] = np.random.uniform(10, 1000, len(train_df))
        if 'slope_mean' not in train_df.columns:
            train_df['slope_mean'] = np.random.uniform(0, 30, len(train_df))
        available_features = ['area', 'distance_to_road', 'slope_mean']
    
    print(f"Model için kullanılan özellikler: {available_features}")
    
    # Eğitim ve test veri setlerini ayır
    X = train_df[available_features]
    y = train_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Özellik ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model oluştur (Random Forest)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Modeli eğit
    rf_model.fit(X_train_scaled, y_train)
    
    # Test seti üzerinde değerlendir
    y_pred = rf_model.predict(X_test_scaled)
    y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Model performansını değerlendir
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Model performansı:")
    print(f"  Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"  Kesinlik (Precision): {precision:.4f}")
    print(f"  Duyarlılık (Recall): {recall:.4f}")
    print(f"  F1 Skoru: {f1:.4f}")
    
    # Özellik önem derecelerini göster
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nÖzellik önem dereceleri:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Tüm parseller için tahmin yap
    X_all = parcels[available_features]
    X_all_scaled = scaler.transform(X_all)
    
    parcels['ml_probability'] = rf_model.predict_proba(X_all_scaled)[:, 1]
    parcels['ml_prediction'] = rf_model.predict(X_all_scaled)
    
    # Model ve scaler'ı dön
    model_artifacts = {
        'model': rf_model,
        'scaler': scaler,
        'features': available_features,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'feature_importance': feature_importance
    }
    
    print("Makine öğrenmesi modeli eğitildi ve tahminler yapıldı.")
    
    return parcels, model_artifacts

def hibrit_model_puanlama(parcels_gdf):
    """
    Kural tabanlı filtreleme ve makine öğrenmesi sonuçlarını birleştiren hibrit bir puanlama sistemi
    """
    print("Hibrit model puanlama sistemi başlatılıyor...")
    
    parcels = parcels_gdf.copy()
    
    # 1. Kural tabanlı filtrelemeye ağırlık ver
    # Eğer herhangi bir kısıtlama varsa ml_probability'yi düşür
    parcels['rule_penalty'] = parcels['restriction_count'] * 0.15
    parcels['adjusted_ml_prob'] = parcels['ml_probability'] * (1 - parcels['rule_penalty'])
    parcels['adjusted_ml_prob'] = parcels['adjusted_ml_prob'].clip(0, 1)
    
    # 2. Zoning_potential ile ml_probability'yi birleştir
    parcels['hybrid_score'] = (
        parcels['zoning_potential'] * 0.4 +  # Özellik mühendisliği skoru
        parcels['adjusted_ml_prob'] * 0.6    # Makine öğrenmesi tahmini (kural penaltısı uygulanmış)
    )
    
    # 3. Kesin kısıtlamaları uygula
    parcels.loc[parcels['has_restrictions'], 'hybrid_score'] *= 0.3
    
    # 4. Son puanlama (0-100 arası)
    parcels['imar_uygunluk_puani'] = (
    parcels['hybrid_score']
    .fillna(0)              # NaN varsa sıfırla
    .replace([np.inf, -np.inf], 0)  # Sonsuz değerleri de sıfırla
    .mul(100)
    .round()
    .astype(int)
    )
    parcels['imar_uygunluk_puani'] = parcels['imar_uygunluk_puani'].clip(0, 100)

    print(parcels[['zoning_potential', 'adjusted_ml_prob', 'has_restrictions', 'hybrid_score', 'imar_uygunluk_puani']].describe())

    
    # 5. Kategorik sınıflandırma
    parcels['imar_uygunluk_sinifi'] = pd.cut(
        parcels['imar_uygunluk_puani'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Çok Yüksek']
    )
    
    # İstatistiksel özet
    sinif_dagilimi = parcels['imar_uygunluk_sinifi'].value_counts()
    sinif_yuzde = parcels['imar_uygunluk_sinifi'].value_counts(normalize=True) * 100
    
    print("\nİmar Uygunluk Sınıfı Dağılımı:")
    for sinif in ['Çok Yüksek', 'Yüksek', 'Orta', 'Düşük', 'Çok Düşük']:
        if sinif in sinif_dagilimi:
            print(f"  {sinif}: {sinif_dagilimi[sinif]} parsel (%{sinif_yuzde[sinif]:.1f})")
    
    print("Hibrit model puanlama tamamlandı.")
    
    return parcels

# ------------ ADIM 5: GÖRSELLEŞTIRME ------------

def gorsellestirme(parcels_gdf, output_dir='output'):
    """
    Model sonuçlarını görselleştirir ve haritalar oluşturur
    """
    print("Görselleştirme başlatılıyor...")
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. İmar Uygunluk Haritası
    print("İmar uygunluk haritası oluşturuluyor...")
    m = folium.Map(
        location=[38.3228, 26.3028],  # Çeşme merkezi
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # Renk paleti
    color_scale = {
        'Çok Düşük': '#FF0000',  # Kırmızı
        'Düşük': '#FFA500',      # Turuncu
        'Orta': '#FFFF00',       # Sarı
        'Yüksek': '#90EE90',     # Açık Yeşil
        'Çok Yüksek': '#008000'  # Koyu Yeşil
    }
    
    # Her parsel için popup bilgisi oluştur
    for _, row in parcels_gdf.iterrows():
        popup_text = f"""
        <b>Parsel ID:</b> {row.get('parcel_id', 'Bilinmiyor')}<br>
        <b>Alan:</b> {row.get('area', 0):.1f} m²<br>
        <b>İmar Uygunluk Puanı:</b> {row.get('imar_uygunluk_puani', 0)}<br>
        <b>İmar Uygunluk Sınıfı:</b> {row.get('imar_uygunluk_sinifi', 'Bilinmiyor')}<br>
        <b>Eğim:</b> {row.get('slope_mean', 0):.1f}°<br>
        <b>Yola Uzaklık:</b> {row.get('distance_to_road', 0):.1f} m
        """
        
        # GeoJSON geometrisini oluştur
        if row.geometry.geom_type == 'Polygon':
            geojson = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {
                    "style": {
                        "color": color_scale.get(row.get('imar_uygunluk_sinifi', 'Çok Düşük')),
                        "fillColor": color_scale.get(row.get('imar_uygunluk_sinifi', 'Çok Düşük')),
                        "fillOpacity": 0.5,
                        "weight": 1
                    }
                }
            }
            
            # GeoJSON katmanını haritaya ekle
            folium.GeoJson(
                geojson,
                style_function=lambda x: x['properties']['style'],
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
    
    # Lejant ekle
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 150px; 
                border:2px solid grey; z-index:9999;
                background-color:white;
                padding: 10px;
                font-size: 14px;">
     <p><b>İmar Uygunluk Sınıfları</b></p>
     <p><i class="fa fa-square" style="color:#008000"></i> Çok Yüksek</p>
     <p><i class="fa fa-square" style="color:#90EE90"></i> Yüksek</p>
     <p><i class="fa fa-square" style="color:#FFFF00"></i> Orta</p>
     <p><i class="fa fa-square" style="color:#FFA500"></i> Düşük</p>
     <p><i class="fa fa-square" style="color:#FF0000"></i> Çok Düşük</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Haritayı kaydet
    m.save(os.path.join(output_dir, 'imar_uygunluk_haritasi.html'))
    
    # 2. İstatistiksel Grafikler
    print("İstatistiksel grafikler oluşturuluyor...")
    plt.figure(figsize=(15, 10))
    
    # 2.1 İmar Uygunluk Sınıfı Dağılımı
    plt.subplot(2, 2, 1)
    sns.countplot(data=parcels_gdf, x='imar_uygunluk_sinifi', 
                 order=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'])
    plt.title('İmar Uygunluk Sınıfı Dağılımı')
    plt.xticks(rotation=45)
    
    # 2.2 Eğim vs İmar Uygunluk Puanı
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=parcels_gdf, x='slope_mean', y='imar_uygunluk_puani', 
                   alpha=0.5, hue='imar_uygunluk_sinifi')
    plt.title('Eğim vs İmar Uygunluk Puanı')
    
    # 2.3 Yola Uzaklık vs İmar Uygunluk Puanı
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=parcels_gdf, x='distance_to_road', y='imar_uygunluk_puani', 
                   alpha=0.5, hue='imar_uygunluk_sinifi')
    plt.title('Yola Uzaklık vs İmar Uygunluk Puanı')
    
    # 2.4 Parsel Alanı vs İmar Uygunluk Puanı
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=parcels_gdf, x='area', y='imar_uygunluk_puani', 
                   alpha=0.5, hue='imar_uygunluk_sinifi')
    plt.title('Parsel Alanı vs İmar Uygunluk Puanı')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'imar_uygunluk_analizi.png'))
    plt.close()
    
    print("Görselleştirme tamamlandı.")
    return True

def rapor_olustur(parcels_gdf, model_artifacts, output_dir='output'):
    """
    Model sonuçlarını ve analizleri içeren detaylı bir rapor oluşturur
    """
    print("Rapor oluşturuluyor...")
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Rapor içeriği
    report = {
        'genel_bilgiler': {
            'toplam_parsel': len(parcels_gdf),
            'ortalama_alan': parcels_gdf['area'].mean(),
            'ortalama_egim': parcels_gdf['slope_mean'].mean(),
            'ortalama_yola_uzaklik': parcels_gdf['distance_to_road'].mean()
        },
        'imar_uygunluk_dagilimi': parcels_gdf['imar_uygunluk_sinifi'].value_counts().to_dict(),
        'kisitlama_analizi': {
            'kisitlamali_parsel_sayisi': parcels_gdf['has_restrictions'].sum(),
            'kisitlama_tipleri': {
                'korunan_alan': parcels_gdf['rule_protected'].sum(),
                'dik_egim': parcels_gdf['rule_steep_slope'].sum(),
                'kucuk_alan': parcels_gdf['rule_small_area'].sum(),
                'yuksek_risk': parcels_gdf['rule_high_geo_risk'].sum(),
                'bozuk_sekil': parcels_gdf['rule_irregular_shape'].sum(),
                'uzak_parsel': parcels_gdf['rule_far_from_road'].sum()
            }
        },
        'model_performansi': model_artifacts.get('metrics', {}),
        'ozellik_onemleri': model_artifacts.get('feature_importance', pd.DataFrame()).to_dict('records')
    }
    
    # Raporu JSON olarak kaydet
    with open(os.path.join(output_dir, 'imar_analiz_raporu.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    # Raporu HTML olarak da oluştur
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>İzmir/Çeşme İmar Analiz Raporu</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ margin-bottom: 15px; }}
            .metric-value {{ font-weight: bold; color: #2980b9; }}
        </style>
    </head>
    <body>
        <h1>İzmir/Çeşme İmar Analiz Raporu</h1>
        
        <h2>Genel Bilgiler</h2>
        <div class="metric">
            <p>Toplam Parsel Sayısı: <span class="metric-value">{report['genel_bilgiler']['toplam_parsel']}</span></p>
            <p>Ortalama Parsel Alanı: <span class="metric-value">{report['genel_bilgiler']['ortalama_alan']:.2f} m²</span></p>
            <p>Ortalama Eğim: <span class="metric-value">{report['genel_bilgiler']['ortalama_egim']:.2f}°</span></p>
            <p>Ortalama Yola Uzaklık: <span class="metric-value">{report['genel_bilgiler']['ortalama_yola_uzaklik']:.2f} m</span></p>
        </div>
        
        <h2>İmar Uygunluk Dağılımı</h2>
        <table>
            <tr>
                <th>Uygunluk Sınıfı</th>
                <th>Parsel Sayısı</th>
                <th>Yüzde</th>
            </tr>
    """
    
    total_parcels = report['genel_bilgiler']['toplam_parsel']
    for sinif, sayi in report['imar_uygunluk_dagilimi'].items():
        yuzde = (sayi / total_parcels) * 100
        html_report += f"""
            <tr>
                <td>{sinif}</td>
                <td>{sayi}</td>
                <td>{yuzde:.1f}%</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Kısıtlama Analizi</h2>
        <table>
            <tr>
                <th>Kısıtlama Tipi</th>
                <th>Parsel Sayısı</th>
            </tr>
    """
    
    for kısıtlama, sayı in report['kisitlama_analizi']['kisitlama_tipleri'].items():
        html_report += f"""
            <tr>
                <td>{kısıtlama.replace('_', ' ').title()}</td>
                <td>{sayı}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Model Performansı</h2>
        <table>
            <tr>
                <th>Metrik</th>
                <th>Değer</th>
            </tr>
    """
    
    for metrik, değer in report['model_performansi'].items():
        html_report += f"""
            <tr>
                <td>{metrik.title()}</td>
                <td>{değer:.4f}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Özellik Önem Dereceleri</h2>
        <table>
            <tr>
                <th>Özellik</th>
                <th>Önem Derecesi</th>
            </tr>
    """
    
    for ozellik in report['ozellik_onemleri']:
        html_report += f"""
            <tr>
                <td>{ozellik['feature']}</td>
                <td>{ozellik['importance']:.4f}</td>
            </tr>
        """
    
    html_report += """
        </table>
    </body>
    </html>
    """
    
    # HTML raporunu kaydet
    with open(os.path.join(output_dir, 'imar_analiz_raporu.html'), 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print("Rapor oluşturma tamamlandı.")
    return True

# ------------ ANA FONKSİYON ------------

def izmir_cesme_imar_modeli_uygula():
    """
    İzmir/Çeşme imara açılabilecek arsa tespit modelini uygular
    """
    print("\n===== İZMİR/ÇEŞME İMARA AÇILABİLECEK ARSA TESPİT MODELİ =====")
    
    # 1. Veri Yükleme
    print("\n----- ADIM 1: VERİ YÜKLEME -----")
    veri_sozlugu = veri_yukle()
    
    # 2. Veri Ön İşleme
    print("\n----- ADIM 2: VERİ ÖN İŞLEME -----")
    islenmis_parceller = veri_on_isleme(veri_sozlugu)
    
    # 3. Özellik Mühendisliği
    print("\n----- ADIM 3: ÖZELLİK MÜHENDİSLİĞİ -----")
    ozellikli_parceller = ozellik_muhendisligi(islenmis_parceller)
    
    # 4. Model Geliştirme
    print("\n----- ADIM 4: MODEL GELİŞTİRME -----")
    
    # 4.1. Kural tabanlı filtreleme
    print("\n4.1. Kural Tabanlı Filtreleme")
    filtrelenmis_parceller = kural_tabanli_filtreleme(ozellikli_parceller)
    
    # 4.2. Makine öğrenmesi modeli
    print("\n4.2. Makine Öğrenmesi Modeli")
    tahminli_parceller, model_artifacts = makine_ogrenmesi_modeli(
        filtrelenmis_parceller, 
        veri_sozlugu.get('training_data', None)
    )
    
    # 4.3. Hibrit model puanlama
    print("\n4.3. Hibrit Model Puanlama")
    puanlanmis_parceller = hibrit_model_puanlama(tahminli_parceller)
    
    # 5. Görselleştirme ve Raporlama
    print("\n----- ADIM 5: GÖRSELLEŞTİRME VE RAPORLAMA -----")
    gorsellestirme(puanlanmis_parceller)
    
    # 6. Rapor oluşturma
    print("\n6. Rapor oluşturma aşaması...")
    rapor_olustur(puanlanmis_parceller, model_artifacts)
    
    # Özet bilgi
    print("\n===== MODEL UYGULAMASI TAMAMLANDI =====")
    print(f"Toplam parsel sayısı: {len(puanlanmis_parceller)}")
    print(f"İmara açılma potansiyeli yüksek parsel sayısı: {len(puanlanmis_parceller[puanlanmis_parceller['imar_uygunluk_sinifi'].isin(['Çok Yüksek', 'Yüksek'])])}")
    print(f"Sınıf dağılımı: {puanlanmis_parceller['imar_uygunluk_sinifi'].value_counts().to_dict()}")
    print(f"Tüm çıktılar '{output_dir}' klasöründe bulunabilir.")
    
    return puanlanmis_parceller, model_artifacts

# Modeli çalıştır
if __name__ == "__main__":
    parceller, model = izmir_cesme_imar_modeli_uygula()
    print("Tamamlandı:", parceller)