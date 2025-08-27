import torch
import cv2
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import DenoisingDataset

# ======= 설정 =======
MODEL_PATH = "/workspace/testcode/results/train/20250813/model_full_20250813_3299.pth"  # 저장된 모델 경로
OUTPUT_DIR = "/workspace/testcode/results/test"                                       # 결과 저장 폴더
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
noisy_dir = '/workspace/testcode/data/can/noised/combined'
clean_dir = '/workspace/testcode/data/can/png'
# Dataset / Dataloader
dataset = DenoisingDataset(noisy_dir, clean_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)


# ======= 모델 로드 =======
print(f"모델 로드 중: {MODEL_PATH}")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()
criterion = nn.MSELoss()
ctn =0
# ======= 추론 수행 =======    
for noisy, clean in loader:
    ctn += 1
    ### test용 ###
    noisy = noisy.to(device)
    clean = clean.to(device)

    with torch.no_grad():
        output_tensor = model(noisy)

    # loss = criterion(output_tensor, clean)

    # ======= 결과 저장 =======
    # basename = path[49:56]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    def to_uint8(img):
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img_norm * 255).astype(np.uint8)
    output_tensor = output_tensor.detach().cpu().squeeze().numpy()
    for i in range(4):
        output_path = os.path.join(OUTPUT_DIR, f"{ctn}_{i}_denoised_{timestamp}.png")
        cv2.imwrite(output_path, to_uint8(output_tensor[i]))
    print(f"추론 완료. 결과 저장됨: {output_path}")
