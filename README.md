# 🧠 Text Detection with U-Net

An advanced deep learning–based text detection system built using a custom U-Net architecture.  
The model detects text regions at pixel level and is trained on the ICDAR 2015 dataset using a hybrid BCE + Dice loss.

---

## ✨ Features

🧠 Custom U-Net architecture for semantic segmentation  
⚖️ Hybrid BCE + Dice loss for better convergence  
🧩 Tile-based processing for large images  
🔄 Data augmentation for robust training  
💾 Checkpoint system with resume capability  
🖼️ Supports JPG, PNG, BMP image formats  
📊 Visualization of masks, heatmaps and detected regions  

---

## 🏗️ Architecture

### Model Structure

**Encoder (Downsampling)**
- DoubleConv (3 → 64)
- MaxPool + DoubleConv (64 → 128)
- MaxPool + DoubleConv (128 → 256)
- MaxPool + DoubleConv (256 → 512)

**Decoder (Upsampling)**
- Upsample + Concat + DoubleConv (512 + 256 → 256)
- Upsample + Concat + DoubleConv (256 + 128 → 128)
- Upsample + Concat + DoubleConv (128 + 64 → 64)

**Output**
- Conv2d (64 → 1)

---

## ⚖️ Loss Function

**BCEDiceLoss**  
A weighted combination of:
- Binary Cross Entropy (pixel-wise classification)
- Dice Loss (region overlap optimization)

Smooth term is used to prevent division by zero.

---

## 📁 Project Structure

- dataset  
  - ch4_training_images  
  - ch4_training_localization_transcription_gt  

- checkpoints  
  - best_model.pth  
  - epoch_XX.pth  

- results  
  - epoch_XX.png  

- test_sonuc  
  - image_name  
    - combined_result.png  
    - region_XX.png  

- Unet.py  
- losses.py  
- dataset2.py  
- train.py  
- test.py  
- visualize.py  

---

## ⚙️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- OpenCV
- NumPy
- Matplotlib
- Shapely

### Setup
Clone the repository and install dependencies using pip.

---

## 📊 Dataset Setup

- Download the ICDAR 2015 dataset  
- Place training images inside `dataset/ch4_training_images`  
- Place ground truth files inside `dataset/ch4_training_localization_transcription_gt`  

---

## 🏃 Training

- Batch Size: 4  
- Tile Size: 512 × 512  
- Stride: 256  
- Learning Rate: 1e-4  
- Epochs: 20  
- Loss: BCE + Dice (0.5 / 0.5)  

Training can be resumed from any saved checkpoint.

---

## 🧪 Testing & Inference

The inference pipeline includes:
- Image tiling and padding
- Tile-wise U-Net inference
- Mask reconstruction
- Thresholding and morphological operations
- Contour detection
- Bounding box extraction
- Cropped text regions

---

## 📈 Evaluation Metrics

- Intersection over Union (IoU)
- Precision
- Recall
- F1-Score
- Dice Coefficient

---

## 🖼️ Results

Detected outputs include:
- Original image
- Prediction heatmap
- Text bounding boxes
- Cropped text regions



## 🛠️ Customization

- Training parameters such as batch size, tile size and learning rate can be adjusted
- BCE / Dice loss weights are configurable
- Detection threshold and minimum area can be tuned for better results

---

## 🚨 Troubleshooting

**CUDA Out of Memory**
- Reduce batch size

**Empty Training Set**
- Check dataset paths
- Verify IoU threshold
- Ensure image and annotation names match

**Poor Detection Performance**
- Increase number of epochs
- Adjust loss weights
- Add more data augmentation

---

## 📚 References

- ICDAR 2015 Text Localization Dataset  
- U-Net: Convolutional Networks for Biomedical Image Segmentation  
- PyTorch Documentation  



        eğitim 
![eğitim görseli ](images/3.png)

        eğitim 
![eğitim görseli ](images/2.png)

        test
![test görseli ](images/1.png)



