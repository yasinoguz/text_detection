import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_results(image, true_mask, pred_mask, save_path=None):
    """Model çıktılarını görselleştirir"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Girdi görüntüsü
    axes[0].imshow(image)
    axes[0].set_title("Girdi Görüntüsü")
    axes[0].axis('off')
    
    # Gerçek maske
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title("Gerçek Maske")
    axes[1].axis('off')
    
    # Tahmin edilen maske
    axes[2].imshow(pred_mask, cmap='gray')
    
    # IoU hesapla ve başlığa ekle
    iou = calculate_iou((pred_mask > 0.5), true_mask)
    axes[2].set_title(f"Tahmin (IoU: {iou:.3f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()