# 🚀 Text Detection with U-Net

An advanced deep learning-based text detection system using U-Net architecture, trained on the ICDAR dataset. This project detects text regions in images with high precision using a combination of BCE and Dice loss functions.

📋 Features
U-Net Architecture: Custom implementation of U-Net for semantic segmentation

Hybrid Loss Function: Combines Binary Cross Entropy (BCE) and Dice Loss for better convergence

Tile-based Processing: Handles large images by splitting them into manageable tiles

Data Augmentation: Built-in transformations for robust training

Checkpoint System: Resume training from any point with full state restoration

Multi-Format Support: Works with JPG, PNG, BMP images

Visualization Tools: Comprehensive result visualization and analysis: Morfolojik işlemler ve kontur analizi


🏗️ Architecture
Model Structure
text
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
Loss Function: BCEDiceLoss
text
BCEDiceLoss = bce_weight × BCEWithLogitsLoss + dice_weight × DiceLoss
- BCE: Handles pixel-wise classification
- Dice: Optimizes for region overlap
- Smooth: Prevents division by zero
📁 Project Structure
text
text-detection-unet/
│
├── dataset/                    # Dataset directory (not included in repo)
│   ├── ch4_training_images/    # Training images
│   └── ch4_training_localization_transcription_gt/  # Ground truth annotations
│
├── checkpoints/                # Model checkpoints
│   ├── best_model.pth
│   └── epoch_XX.pth
│
├── results/                    # Training results and visualizations
│   └── epoch_XX.png
│
├── test_sonuc/                 # Test results with detected regions
│   └── {image_name}/
│       ├── combined_result.png
│       └── region_XX.png
│
├── Unet.py                     # U-Net model implementation
├── losses.py                   # Loss functions (BCEDiceLoss)
├── dataset2.py                 # Dataset loader and tile processing
├── train.py                    # Training script with resume capability
├── test.py                     # Testing and inference script
└── visualize.py                # Visualization utilities



🚀 Installation
Prerequisites
Python 3.8+

PyTorch 1.9+

OpenCV

NumPy

Matplotlib


Install Dependencies
bash
# Clone the repository
git clone https://github.com/yourusername/text-detection-unet.git
cd text-detection-unet

# Install required packages
pip install torch torchvision opencv-python numpy matplotlib shapely tqdm
Dataset Setup
Download the ICDAR 2015 dataset

Place the training images in dataset/ch4_training_images/

Place the ground truth text files in dataset/ch4_training_localization_transcription_gt/

🏃‍♂️ Usage
Training
bash
python train.py
Training Configuration:

Batch Size: 4

Tile Size: 512×512

Stride: 256

Learning Rate: 1e-4

Epochs: 20

Loss: BCEDiceLoss (BCE weight: 0.5, Dice weight: 0.5)


Resume Training:

python
# In train.py, set the resume path:
resume_pth = "checkpoints/epoch_05.pth"
Testing/Inference
bash
python test.py
Before running, update in test.py:

python
MODEL_PATH = "checkpoints/epoch_05.pth"  # Path to your model checkpoint
IMAGE_PATH = "path/to/your/image.png"    # Path to test image
📊 Dataset Processing
Tile Generation
Images are split into overlapping tiles (512×512)

Stride of 256 ensures coverage while maintaining context

Only tiles containing text (IOU > threshold) are used for training

Padding is applied to handle edge cases


Mask Creation
Polygons are converted to binary masks

Each tile gets its corresponding mask for supervised training

🧠 Model Details
DoubleConv Block
Each convolutional block contains:

Conv2d (3×3 kernel, padding=1)

Batch Normalization

ReLU Activation

Conv2d (3×3 kernel, padding=1)

Batch Normalization

ReLU Activation

Training Features
Learning Rate Scheduling: ReduceLROnPlateau with factor=0.5, patience=2

Checkpointing: Saves model after each epoch

Best Model Tracking: Automatically saves the best model based on validation loss

Visualization: Saves sample predictions after each epoch

Resume Capability: Can continue training from any checkpoint

📈 Performance Metrics
Loss Function
BCE Loss: Measures pixel-wise classification error

Dice Loss: Measures region overlap (1 - Dice coefficient)

Total Loss: Weighted combination of both losses

Evaluation Metrics
Intersection over Union (IoU): For segmentation quality

Precision/Recall: For text detection accuracy

F1-Score: Balance between precision and recall

🎯 Inference Pipeline
Step-by-Step Process:
Image Preprocessing

Read and convert to RGB

Pad to make dimensions divisible by tile size

Split into overlapping tiles

Model Inference

Process each tile through U-Net

Apply sigmoid activation

Generate probability maps

Post-processing

Reconstruct full image from tile predictions

Apply weighted averaging for smooth transitions

Threshold to create binary mask

Apply morphological operations

Text Region Detection

Find contours in binary mask

Filter by area and aspect ratio

Extract bounding boxes

Save individual text regions

🛠️ Customization
Modify Training Parameters
python
# In train.py
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
TILE_SIZE = 640
STRIDE = 320
Custom Loss Weights
python
# In train.py
criterion = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
Adjust Text Detection Thresholds
python
# In test.py
threshold = 0.5  # Binary threshold
min_area = 50    # Minimum text region area
📝 Results
Output Structure
After testing, results are saved in test_sonuc/{image_name}/:

combined_result.png: Side-by-side comparison

region_XX.png: Individual detected text regions

Visualization
The system provides:

Original image

Prediction heatmap

Detected text regions with bounding boxes

Individual cropped text regions

🔧 Troubleshooting
Common Issues
CUDA Out of Memory

python
# Reduce batch size in train.py
dataloader = DataLoader(dataset, batch_size=2, ...)
Empty Training Set

Check dataset paths

Verify IOU threshold in dataset2.py

Ensure text files match image files

Poor Detection Results

Increase training epochs

Adjust loss weights

Add data augmentation

Try different threshold values

Slow Inference

Increase tile stride

Reduce image resolution

Use GPU acceleration

📚 References
U-Net: Convolutional Networks for Biomedical Image Segmentation

ICDAR 2015 Dataset

PyTorch Documentation



        eğitim 
![eğitim görseli ](images/3.png)

        eğitim 
![eğitim görseli ](images/2.png)

        test
![test görseli ](images/1.png)



