# 📝 Text Detection with U-Net

---

An advanced deep learning-based text detection system using **U-Net architecture**, trained on the **ICDAR dataset**.  
This project detects text regions in images with high precision using a combination of **BCE** and **Dice loss functions**.

---

## 📋 Features

- 🧠 **U-Net Architecture**: Custom implementation of U-Net for semantic segmentation  
- ⚖️ **Hybrid Loss Function**: Combines Binary Cross Entropy (BCE) and Dice Loss for better convergence  
- 🧩 **Tile-based Processing**: Handles large images by splitting them into manageable tiles  
- 🔄 **Data Augmentation**: Built-in transformations for robust training  
- 💾 **Checkpoint System**: Resume training from any point with full state restoration  
- 🖼️ **Multi-Format Support**: Works with JPG, PNG, BMP images  
- 📊 **Visualization Tools**: Comprehensive result visualization and analysis  

---

## 🏗️ Architecture

### 🔧 Model Structure

```text
UNet Architecture:
├── Encoder (Downsampling)
│   ├── DoubleConv(3 → 64)
│   ├── MaxPool + DoubleConv(64 → 128)
│   ├── MaxPool + DoubleConv(128 → 256)
│   └── MaxPool + DoubleConv(256 → 512)
│
├── Decoder (Upsampling)
│   ├── Upsample + Concat + DoubleConv(512+256 → 256)
│   ├── Upsample + Concat + DoubleConv(256+128 → 128)
│   └── Upsample + Concat + DoubleConv(128+64 → 64)
│
└── Output Layer
    └── Conv2d(64 → 1)
⚖️ Loss Function: BCEDiceLoss
text
Kodu kopyala
BCEDiceLoss = bce_weight × BCEWithLogitsLoss + dice_weight × DiceLoss
🟦 BCE: Handles pixel-wise classification

🟩 Dice: Optimizes for region overlap

🧮 Smooth: Prevents division by zero

📁 Project Structure
text
Kodu kopyala
text-detection-unet/
│
├── dataset/                    
│   ├── ch4_training_images/    
│   └── ch4_training_localization_transcription_gt/  
│
├── checkpoints/                
│   ├── best_model.pth
│   └── epoch_XX.pth
│
├── results/                    
│   └── epoch_XX.png
│
├── test_sonuc/                 
│   └── {image_name}/
│       ├── combined_result.png
│       └── region_XX.png
│
├── Unet.py                     
├── losses.py                   
├── dataset2.py                 
├── train.py                    
├── test.py                     
└── visualize.py                
🚀 Installation
✅ Prerequisites
Python 3.8+

PyTorch 1.9+

OpenCV

NumPy

Matplotlib



📦 Install Dependencies
bash
Kodu kopyala
# Clone the repository
git clone https://github.com/yourusername/text-detection-unet.git
cd text-detection-unet

# Install required packages
pip install torch torchvision opencv-python numpy matplotlib shapely tqdm
📂 Dataset Setup
Download the ICDAR 2015 dataset

Place training images in:

awk
Kodu kopyala
dataset/ch4_training_images/
Place ground truth files in:

awk
Kodu kopyala
dataset/ch4_training_localization_transcription_gt/
🏃‍♂️ Usage
🎯 Training
bash
Kodu kopyala
python train.py
Training Configuration:

Batch Size: 4

Tile Size: 512×512

Stride: 256

Learning Rate: 1e-4

Epochs: 20

Loss: BCEDiceLoss (BCE weight: 0.5, Dice weight: 0.5)

🔁 Resume Training
python
Kodu kopyala
resume_pth = "checkpoints/epoch_05.pth"
🔍 Testing / Inference
bash
Kodu kopyala
python test.py
Before running, update in test.py:

python
Kodu kopyala
MODEL_PATH = "checkpoints/epoch_05.pth"
IMAGE_PATH = "path/to/your/image.png"
📊 Dataset Processing
🧩 Tile Generation
Images are split into overlapping tiles (512×512)

Stride of 256 ensures coverage

Only tiles containing text (IOU > threshold) are used

Padding is applied for edge cases

🧾 Annotation Format
text
Kodu kopyala
x1,y1,x2,y2,x3,y3,x4,y4,text
🎭 Mask Creation
Polygons are converted to binary masks

Each tile gets its corresponding mask

🧠 Model Details
🔹 DoubleConv Block
Each block contains:

Conv2d (3×3 kernel, padding=1)

Batch Normalization

ReLU Activation

Conv2d (3×3 kernel, padding=1)

Batch Normalization

ReLU Activation

📈 Performance Metrics
📉 Loss Function
BCE Loss

Dice Loss

Total Loss (weighted combination)

📐 Evaluation Metrics
IoU

Precision / Recall

F1-Score

🎯 Inference Pipeline
🖼️ Image Preprocessing

🧠 Model Inference

🧹 Post-processing

📦 Text Region Detection

🛠️ Customization
🔧 Modify Training Parameters
python
Kodu kopyala
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
TILE_SIZE = 640
STRIDE = 320
⚖️ Custom Loss Weights
python
Kodu kopyala
criterion = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
🎚️ Adjust Detection Thresholds
python
Kodu kopyala
threshold = 0.5
min_area = 50
📝 Results
combined_result.png

region_XX.png

Includes:

Original image

Prediction heatmap

Detected text regions

Cropped text regions

🔧 Troubleshooting
❌ CUDA Out of Memory
python
Kodu kopyala
dataloader = DataLoader(dataset, batch_size=2, ...)
❌ Empty Training Set
Check dataset paths

Verify IOU threshold

Match image & text files

❌ Poor Detection Results
Increase epochs

Adjust loss weights

Add augmentation

📚 References

ICDAR 2015 Dataset

PyTorch Documentation



        eğitim 
![eğitim görseli ](images/3.png)

        eğitim 
![eğitim görseli ](images/2.png)

        test
![test görseli ](images/1.png)



