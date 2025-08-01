import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from dataset2 import ICDARTileDataset, list_images, list_txts
from Unet import UNet
from losses import BCEDiceLoss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_DIR = "../dataset/ch4_training_images"
    TXT_DIR = "../dataset/ch4_training_localization_transcription_gt"
    RESULTS_DIR = "results"
    check_point_dir="checkpoints"
    CHECKPOINT_PATH = os.path.join(check_point_dir, "best_model.pth")
    
    # ===== RESUME ÖZELLİĞİ =====
    resume_pth = "checkpoints/epoch_05.pth"  # Buraya pth dosya yolunu girin (örn: "checkpoints/epoch_05.pth")
    # ============================
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(check_point_dir, exist_ok=True)  # Checkpoints klasörünü oluştur

    img_paths = list_images(IMG_DIR)
    txt_paths = list_txts(TXT_DIR)
    dataset = ICDARTileDataset(img_paths, txt_paths, tile_size=512, stride=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    NUM_EPOCHS = 20
    best_loss = float('inf')
    start_epoch = 0  # Resume özelliği tarafından güncellenecek

    # Resume özelliği - eğer resume_pth dosyası varsa oradan devam et
    if resume_pth and os.path.exists(resume_pth):
        print(f"🔄 Resume dosyası bulundu: {resume_pth}")
        checkpoint = torch.load(resume_pth, map_location=device)
        
        # Eski format (sadece model state dict) veya yeni format (checkpoint dict) kontrolü
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Yeni format - checkpoint dictionary
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            print(f"✅ Training {start_epoch}. epoch'tan devam ediyor...")
        else:
            # Eski format - sadece model state dict
            model.load_state_dict(checkpoint)
            print("✅ Eski format model yüklendi, sıfırdan başlıyor...")
    else:
        print("🔄 Sıfırdan training başlıyor...")

    print(f"Dataset tile sayısı: {len(dataset)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Batch sayısı (len(dataloader)): {len(dataloader)}")
    print(f"Toplam örnek sayısı (batch sayısı * batch size): {len(dataloader)*dataloader.batch_size}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # Her epoch sonunda modeli kaydet
        epoch_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'loss': avg_loss
        }
        epoch_path = os.path.join(check_point_dir, f"epoch_{epoch+1:02d}.pth")
        torch.save(epoch_checkpoint, epoch_path)
        print(f"💾 Epoch {epoch+1} checkpoint kaydedildi: {epoch_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("✅ Model kaydedildi (best_loss güncellendi).")

        model.eval()
        with torch.no_grad():
            # Rastgele bir örnek seç
            idx = random.randint(0, len(dataset) - 1)
            sample_img, sample_mask = dataset[idx]
            sample_img = sample_img.unsqueeze(0).to(device)
            pred = model(sample_img)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_img = (pred > 0.5).astype("uint8") * 255

            orig_img = (sample_img.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
            mask_np = (sample_mask.squeeze().cpu().numpy() * 255).astype("uint8")

            vis = np.hstack([
                orig_img,
                cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
            ])
            out_path = os.path.join(RESULTS_DIR, f"epoch_{epoch+1:02d}.png")
            cv2.imwrite(out_path, vis)

if __name__ == '__main__':
    train()
