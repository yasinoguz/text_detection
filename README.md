# 🧠 Text Detection with U-Net (ICDAR 2015)

A deep learning–based **text detection system** built with a custom **U-Net architecture**.  
The model performs **pixel-level text segmentation** and is trained on the **ICDAR 2015 dataset** using a hybrid **BCE + Dice loss** for high-precision text localization.

---

## ✨ Key Features

✅ Custom U-Net Architecture for semantic segmentation  
✅ Hybrid Loss Function (Binary Cross Entropy + Dice Loss)  
✅ Tile-based Processing for large images  
✅ Data Augmentation for robust training  
✅ Checkpoint & Resume Training support  
✅ Multi-format Image Support (JPG · PNG · BMP)  
✅ Visualization tools (heatmaps, masks, bounding boxes, cropped regions)

---

## 🏗️ Model Architecture

### 🔹 U-Net Structure

Encoder (Downsampling)
├── DoubleConv(3 → 64)
├── MaxPool + DoubleConv(64 → 128)
├── MaxPool + DoubleConv(128 → 256)
└── MaxPool + DoubleConv(256 → 512)

Decoder (Upsampling)
├── Upsample + Concat + DoubleConv(512+256 → 256)
├── Upsample + Concat + DoubleConv(256+128 → 128)
└── Upsample + Concat + DoubleConv(128+64 → 64)

Output Layer
└── Conv2d(64 → 1)

yaml
Kodu kopyala

---

## 🧪 Loss Function

### 🔹 BCEDiceLoss

Loss = α × BCEWithLogitsLoss + β × DiceLoss

yaml
Kodu kopyala

- **BCE Loss**: Pixel-wise classification  
- **Dice Loss**: Region overlap optimization  
- **Smooth term** prevents division by zero  

---

## 📁 Project Structure

text-detection-unet/
│
├── dataset/ # Dataset (not included)
│ ├── ch4_training_images/
│ └── ch4_training_localization_transcription_gt/
│
├── checkpoints/
│ ├── best_model.pth
│ └── epoch_XX.pth
│
├── results/
│ └── epoch_XX.png
│
├── test_sonuc/
│ └── {image_name}/
│ ├── combined_result.png
│ └── region_XX.png
│
├── Unet.py # U-Net model
├── losses.py # BCEDiceLoss
├── dataset2.py # Dataset & tiling
├── train.py # Training script
├── test.py # Inference script
└── visualize.py # Visualization tools

yaml
Kodu kopyala

---

## ⚙️ Installation

### 🔹 Requirements

- Python 3.8+
- PyTorch 1.9+
- OpenCV
- NumPy
- Matplotlib
- Shapely

### 🔹 Setup

git clone https://github.com/yourusername/text-detection-unet.git
cd text-detection-unet
pip install torch torchvision opencv-python numpy matplotlib shapely tqdm

yaml
Kodu kopyala

---

## 📊 Dataset Setup

1. Download **ICDAR 2015**  
2. Place training images:
dataset/ch4_training_images/

markdown
Kodu kopyala
3. Place ground truth files:
dataset/ch4_training_localization_transcription_gt/

yaml
Kodu kopyala

---

## 🏃 Training

python train.py

nix
Kodu kopyala

### 🔧 Default Training Settings

- Batch Size: 4  
- Tile Size: 512×512  
- Stride: 256  
- Learning Rate: 1e-4  
- Epochs: 20  
- Loss: BCE + Dice (0.5 / 0.5)

### 🔁 Resume Training

resume_pth = "checkpoints/epoch_05.pth"

yaml
Kodu kopyala

---

## 🧪 Testing / Inference

python test.py

n1ql
Kodu kopyala

Update in `test.py`:

MODEL_PATH = "checkpoints/epoch_05.pth"
IMAGE_PATH = "path/to/image.png"

yaml
Kodu kopyala

---

## 🔍 Inference Pipeline

1. Image tiling & padding  
2. U-Net inference per tile  
3. Probability map reconstruction  
4. Thresholding + morphological operations  
5. Contour detection  
6. Bounding box extraction  
7. Cropped text regions  

---

## 📈 Evaluation Metrics

- Intersection over Union (IoU)  
- Precision / Recall  
- F1-Score  
- Dice Coefficient  

---

## 🖼️ Output Results

test_sonuc/{image_name}/
├── combined_result.png
├── region_01.png
├── region_02.png

yaml
Kodu kopyala

Includes:
- Original image  
- Prediction heatmap  
- Detected text bounding boxes  
- Cropped text regions  

---

## 🛠️ Customization

### 🔹 Loss Weights

criterion = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)

clean
Kodu kopyala

### 🔹 Detection Thresholds

threshold = 0.5
min_area = 50

yaml
Kodu kopyala

---

## 🚨 Troubleshooting

**CUDA Out of Memory**  
- Reduce batch size  

**Empty Training Set**  
- Check dataset paths  
- Verify IoU threshold  
- Ensure GT files match images  

**Poor Detection Results**  
- Increase epochs  
- Adjust loss weights  
- Tune threshold values  

---

## 📚 References

- U-Net: Convolutional Networks for Biomedical Image Segmentation  
- ICDAR 2015 Dataset  
- PyTorch Documentation  


        eğitim 
![eğitim görseli ](images/3.png)

        eğitim 
![eğitim görseli ](images/2.png)

        test
![test görseli ](images/1.png)



