import os
import cv2
import torch
import numpy as np
from Unet import UNet
import matplotlib.pyplot as plt

def load_model(model_path):
    """Modeli yükler ve hazırlar"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    model = UNet(n_channels=3, n_classes=1)
    
    # Checkpoint dosyasını yükle
    checkpoint = torch.load(model_path, map_location=device)
    
    # Checkpoint formatını kontrol et
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint formatında kaydedilmiş model
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint formatında model yüklendi (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        # Sadece model state dict formatında
        model.load_state_dict(checkpoint)
        print("✅ Model state dict formatında yüklendi")
    
    model.to(device)
    model.eval()
    print("✅ Model başarıyla yüklendi!")
    return model, device

def preprocess_image(image_path, tile_size=512, stride=256):
    """Görüntüyü tile'lara böler ve ön işleme yapar"""
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    
    # BGR'den RGB'ye çevir
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    original_h, original_w = img.shape[:2]
    print(f"Orijinal görüntü boyutu: {original_w}x{original_h}")
    
    # Görüntüyü tile_size'ın katı yapmak için padding ekle
    pad_h = (tile_size - original_h % tile_size) % tile_size
    pad_w = (tile_size - original_w % tile_size) % tile_size
    
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        print(f"Padding sonrası boyut: {img.shape[1]}x{img.shape[0]}")
    
    h, w = img.shape[:2]
    
    # Tile'ları oluştur
    tiles = []
    tile_positions = []
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            tile_positions.append((x, y))
    
    print(f"Oluşturulan tile sayısı: {len(tiles)}")
    return tiles, tile_positions, (original_h, original_w, 3)  # Orijinal boyutu döndür

def predict_tiles(model, tiles, device):
    """Tile'ları model ile tahmin eder"""
    predictions = []
    
    with torch.no_grad():
        for i, tile in enumerate(tiles):
            # Tile'ı tensor'a çevir ve normalize et
            tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float() / 255.0
            tile_tensor = tile_tensor.unsqueeze(0).to(device)
            
            # Tahmin yap
            pred = model(tile_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            predictions.append(pred)
            
            if i % 10 == 0:
                print(f"✅ Tahmin edilen tile: {i+1}/{len(tiles)}")
    
    return predictions

def reconstruct_predictions(predictions, tile_positions, original_shape, tile_size=512, stride=256):
    """Tile tahminlerini orijinal görüntü boyutunda birleştirir"""
    original_h, original_w = original_shape[:2]
    full_pred = np.zeros((original_h, original_w), dtype=np.float32)
    count_map = np.zeros((original_h, original_w), dtype=np.float32)
    
    # Basit ağırlık (merkeze yakın piksellere daha fazla ağırlık)
    weight = np.ones((tile_size, tile_size))
    center = tile_size // 2
    for i in range(tile_size):
        for j in range(tile_size):
            # Basit mesafe ağırlığı
            dist = np.sqrt((i-center)**2 + (j-center)**2)
            weight[i, j] = np.exp(-dist / (center * 0.5))
    
    for pred, (x, y) in zip(predictions, tile_positions):
        # Sınırları kontrol et (orijinal boyuta göre)
        end_y = min(y + tile_size, original_h)
        end_x = min(x + tile_size, original_w)
        
        # Tahmin boyutunu kontrol et
        if pred.shape != (tile_size, tile_size):
            pred = cv2.resize(pred, (tile_size, tile_size))
        
        # Kırpılmış ağırlık ve tahmin
        cropped_weight = weight[:end_y-y, :end_x-x]
        cropped_pred = pred[:end_y-y, :end_x-x]
        
        # Güvenli birleştirme
        try:
            full_pred[y:end_y, x:end_x] += cropped_pred * cropped_weight
            count_map[y:end_y, x:end_x] += cropped_weight
        except ValueError as e:
            print(f"⚠️ Birleştirme hatası: {e}")
            print(f"   Tahmin boyutu: {cropped_pred.shape}")
            print(f"   Ağırlık boyutu: {cropped_weight.shape}")
            print(f"   Hedef alan: {end_y-y}x{end_x-x}")
            # Basit toplama yap
            full_pred[y:end_y, x:end_x] += cropped_pred
            count_map[y:end_y, x:end_x] += 1
    
    # Ağırlıklı ortalama al
    full_pred = full_pred / (count_map + 1e-8)
    
    return full_pred

def detect_text_regions(prediction, threshold=0.5, min_area=100):
    """Tahmin sonucundan yazı bölgelerini tespit eder"""
    # Binary mask oluştur
    binary_mask = (prediction > threshold).astype(np.uint8) * 255
    
    # Geliştirilmiş morfolojik işlemler
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Konturları bul
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gelişmiş filtreleme
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # En-boy oranı filtresi
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.1 or aspect_ratio > 10:  # Aşırı dar/uzun şekilleri ele
            continue
            
        valid_contours.append(contour)
    
    return valid_contours, binary_mask

def visualize_results(original_img, prediction, contours, save_path=None, image_name="test_image"):
    """Sonuçları görselleştirir ve her tespit edilen yazı bölgesini ayrı kaydeder"""
    # Orijinal görüntüyü BGR'ye çevir (OpenCV için)
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    else:
        original_bgr = original_img
    
    # Tahmin maskesini görselleştir
    pred_vis = (prediction * 255).astype(np.uint8)
    pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
    
    # Konturları çiz
    result_img = original_bgr.copy()
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
    
    # Her tespit edilen yazı bölgesini ayrı kaydet
    detected_regions = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Yazı bölgesini kes
        region = original_bgr[y:y+h, x:x+w]
        detected_regions.append((region, (x, y, w, h), i))
    
    # Sonuçları birleştir
    h, w = original_bgr.shape[:2]
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, :w] = original_bgr
    combined[:, w:w*2] = pred_vis
    combined[:, w*2:] = result_img
    
    # Görselleştir
    plt.figure(figsize=(18, 6))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal | Tahmin Maskesi | Yazı Tespiti")
    plt.axis('off')
    
    if save_path:
        # Dosya adından uzantıyı çıkar ve klasör oluştur
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        result_folder = f"test_sonuc/{base_name}"
        os.makedirs(result_folder, exist_ok=True)
        
        # Ana sonuç resmini kaydet
        combined_filename = f"{result_folder}/combined_result.png"
        cv2.imwrite(combined_filename, combined)
        print(f"Ana sonuç kaydedildi: {combined_filename}")
        
        # Her tespit edilen yazı bölgesini ayrı kaydet
        if detected_regions:
            for region, (x, y, w, h), idx in detected_regions:
                region_filename = f"{result_folder}/region_{idx+1:02d}.png"
                cv2.imwrite(region_filename, region)
                print(f"Yazı bölgesi {idx+1} kaydedildi: {region_filename} (koordinat: x={x}, y={y}, w={w}, h={h})")
    
    plt.show()
    
    return result_img, detected_regions

def test_model():
    """Ana test fonksiyonu"""
    # ===== DOSYA YOLLARINI BURAYA GİRİN =====
    MODEL_PATH = "checkpoints/epoch_05.pth"  # best_model.pth dosyasının yolu
    IMAGE_PATH = "../dataset/fis11.png"  # Test edilecek görüntünün yolu
    # =========================================
    
    # Results klasörünü oluştur
    os.makedirs("results", exist_ok=True)
    
    if not MODEL_PATH or not IMAGE_PATH:
        print("❌ Lütfen MODEL_PATH ve IMAGE_PATH değerlerini doldurun!")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        return
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Görüntü dosyası bulunamadı: {IMAGE_PATH}")
        return
    
    try:
        # Modeli yükle
        print("🔄 Model yükleniyor...")
        model, device = load_model(MODEL_PATH)
        
        print(f"📏 Görüntü boyutu kontrol ediliyor: {IMAGE_PATH}")
        
        # Görüntüyü işle
        print("🔄 Görüntü işleniyor...")
        tiles, tile_positions, original_shape = preprocess_image(IMAGE_PATH)
        
        # Tahmin yap
        print("🔄 Tahmin yapılıyor...")
        predictions = predict_tiles(model, tiles, device)
        
        # Sonuçları birleştir
        print("🔄 Sonuçlar birleştiriliyor...")
        full_prediction = reconstruct_predictions(predictions, tile_positions, original_shape)
        
        # Yazı bölgelerini tespit et
        print("🔄 Yazı bölgeleri tespit ediliyor...")
        contours, binary_mask = detect_text_regions(full_prediction)
        
        # Orijinal görüntüyü oku (padding olmadan)
        original_img = cv2.imread(IMAGE_PATH)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Tahmin sonucunu orijinal boyuta kırp
        pred_h, pred_w = full_prediction.shape
        orig_h, orig_w = original_img.shape[:2]
        
        if pred_h != orig_h or pred_w != orig_w:
            print(f"⚠️ Tahmin boyutu ({pred_w}x{pred_h}) orijinal boyut ({orig_w}x{orig_h}) ile uyumsuz")
            print("Tahmin sonucu orijinal boyuta kırpılıyor...")
            full_prediction = full_prediction[:orig_h, :orig_w]
        
        # Sonuçları görselleştir
        print("🔄 Sonuçlar görselleştiriliyor...")
        result_img, detected_regions = visualize_results(original_img, full_prediction, contours, 
                                                       save_path="results", 
                                                       image_name=IMAGE_PATH)
        
        print(f"✅ Test tamamlandı! {len(contours)} yazı bölgesi tespit edildi.")
        print(f"📁 Tespit edilen yazı bölgeleri 'results' klasörüne kaydedildi.")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    test_model()