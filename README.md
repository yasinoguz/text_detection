# U-Net ile Metin Tespiti (Text Detection)


Bu proje, ICDAR veri seti kullanılarak eğitilmiş bir U-Net modeliyle görüntülerdeki metin bölgelerinin tespitini yapar. Özellikle belge/fotoğraf işleme ve OCR öncesi segmentasyon için tasarlanmıştır.

## Öne Çıkan Özellikler
- **Optimize U-Net Modeli**: 4 katmanlı encoder-decoder yapısı
- **Akıllı Kayıp Fonksiyonu**: BCE + Dice Loss kombinasyonu
- **Büyük Görüntü Desteği**: 512x512 tile'lar ile işlem yapabilme
- **Eğitim Esnekliği**: Yarıda kalan eğitime devam edebilme
- **Gelişmiş Post-Processing**: Morfolojik işlemler ve kontur analizi

## Kurulum
Gereksinimler:
pip install torch torchvision opencv-python numpy matplotlib shapely tqdm

Kullanım
Eğitim
python
python train.py
Parametreler:

resume_pth: Yarıda kalan eğitime devam etmek için checkpoint yolu

tile_size: Görüntü tile boyutu (default: 512)

stride: Tile kaydırma mesafesi (default: 256)


Veri Seti
ICDAR 2015 veri seti kullanılmıştır:

1000+ eğitim görseli

Metin koordinatları için .txt annotasyonları

Çeşitli diller ve metin yönelimleri





        eğitim 
![eğitim görseli ](images/3.png)

        eğitim 
![eğitim görseli ](images/2.png)

        test
![test görseli ](images/1.png)



