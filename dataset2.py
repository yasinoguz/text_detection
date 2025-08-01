import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def read_polygons(txt_path):
    polys = []
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(',')
            coords = list(map(float, parts[:8]))
            poly = np.array(coords).reshape((4, 2))
            polys.append(poly)
    return polys


def polygon_iou(poly1, poly2):
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    if not p1.is_valid or not p2.is_valid:
        return 0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    if union == 0:
        return 0
    return inter / union


class ICDARTileDataset(Dataset):
    def __init__(self, img_paths, txt_paths, tile_size=512, stride=256, transforms=None, iou_thresh=0):
        self.img_paths = img_paths
        self.txt_paths = txt_paths
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms
        self.iou_thresh = iou_thresh  # IOU eşik değeri, default biraz düşük tutuldu

        self.tiles = []  # (img_idx, x, y) tupleları
        self._prepare_tiles()

    def _prepare_tiles(self):
        total_tiles = 0
        selected_tiles = 0
        for idx, (img_path, txt_path) in enumerate(zip(self.img_paths, self.txt_paths)):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Image not found or can't be read: {img_path}")
                continue
            h, w = img.shape[:2]
            polys = read_polygons(txt_path)
            for y in range(0, h - self.tile_size + 1, self.stride):
                for x in range(0, w - self.tile_size + 1, self.stride):
                    total_tiles += 1
                    tile_box = np.array([[x, y], [x + self.tile_size, y], [x + self.tile_size, y + self.tile_size], [x, y + self.tile_size]])
                    has_text = False
                    for poly in polys:
                        if polygon_iou(tile_box, poly) > self.iou_thresh:  # IOU eşiği artık parametre
                            has_text = True
                            break
                    if has_text:
                        selected_tiles += 1
                        self.tiles.append((idx, x, y))

        print(f"Toplam tile sayısı: {total_tiles}")
        print(f"Seçilen tile sayısı (has_text=True): {selected_tiles}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_idx, x, y = self.tiles[idx]
        img = cv2.imread(self.img_paths[img_idx])
        tile_img = img[y:y + self.tile_size, x:x + self.tile_size]

        polys = read_polygons(self.txt_paths[img_idx])
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        for poly in polys:
            poly_in_tile = poly - np.array([x, y])
            poly_in_tile = poly_in_tile.astype(np.int32)
            cv2.fillPoly(mask, [poly_in_tile], 255)

        if self.transforms:
            tile_img = self.transforms(tile_img)

        tile_img = torch.from_numpy(tile_img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask / 255.0).float().unsqueeze(0)

        return tile_img, mask


def list_images(img_dir):
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = []
    for fname in sorted(os.listdir(img_dir)):
        if any(fname.lower().endswith(ext) for ext in exts):
            img_paths.append(os.path.join(img_dir, fname))
    return img_paths


def list_txts(txt_dir):
    txt_paths = []
    for fname in sorted(os.listdir(txt_dir)):
        if fname.lower().endswith('.txt'):
            txt_paths.append(os.path.join(txt_dir, fname))
    return txt_paths


def show_tile_and_mask(dataset, idx):
    tile_img, mask = dataset[idx]
    img_np = (tile_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    mask_np = (mask.squeeze(0).numpy() * 255).astype(np.uint8)

    # Maskeyi kırmızı renkte overlay yap
    overlay = img_np.copy().astype(np.float32)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_np * 0.5, 0, 255)  # Kırmızı kanal

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("Tile Image")
    axs[0].axis('off')

    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title("Mask")
    axs[1].axis('off')

    axs[2].imshow(overlay.astype(np.uint8))
    axs[2].set_title("Overlay")
    axs[2].axis('off')

    plt.show()


# if __name__ == "__main__":
#      img_dir = "../dataset/ch4_training_images"  # Kendi yolunuza göre değiştirin
#      txt_dir = "../dataset/ch4_training_localization_transcription_gt"  # Kendi yolunuza göre değiştirin

#      img_paths = list_images(img_dir)
#      txt_paths = list_txts(txt_dir)
#      print(f"Found {len(img_paths)} images and {len(txt_paths)} txt files")

#      dataset = ICDARTileDataset(img_paths, txt_paths, tile_size=512, stride=256)
#      print(f"Dataset prepared with {len(dataset)} tiles containing text.")

#      # Örnek gösterim
#      how_tile_and_mask(dataset, 0)
