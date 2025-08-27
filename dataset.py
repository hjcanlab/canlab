import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T

def force_to_480x640(img_np):
    """
    입력: numpy array (H, W)
    출력: (480, 640)로 강제 변환 (crop 또는 pad 포함)
    """
    target_h, target_w = 480, 640
    H, W = img_np.shape

    # 1. 자르기
    if H > target_h:
        crop_top = (H - target_h) // 2
        img_np = img_np[crop_top:crop_top + target_h, :]
        H = target_h
    if W > target_w:
        crop_left = (W - target_w) // 2
        img_np = img_np[:, crop_left:crop_left + target_w]
        W = target_w

    # 2. 패딩
    pad_h = max(target_h - H, 0)
    pad_w = max(target_w - W, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return cv2.copyMakeBorder(
        img_np,
        pad_top, pad_bottom,
        pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )


class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

        self.noisy_files = sorted(os.listdir(noisy_dir))
        self.clean_files = sorted(os.listdir(clean_dir))

        assert len(self.noisy_files) == len(self.clean_files), "Noisy와 Clean 이미지 개수가 다릅니다."

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy = Image.open(noisy_path).convert('L')  # 그레이스케일
        clean = Image.open(clean_path).convert('L')
        noisy_np = np.array(noisy)  # (H, W)
        clean_np = np.array(clean)
        H, W = noisy_np.shape
        if H == 480 and W == 640:
            # print(f"print_H : {H}, W : {W}")
            pass
        else:
            print(f"print_H : {H}, W : {W}")
            noisy_np = force_to_480x640(noisy_np)
            clean_np = force_to_480x640(clean_np)

        noisy = Image.fromarray(noisy_np)
        clean = Image.fromarray(clean_np)

        transform = T.ToTensor()
        noisy = transform(noisy)
        clean = transform(clean)

        # assert
        assert noisy.shape == clean.shape == (1, 480, 640), \
            f"Shape mismatch! {noisy.shape} vs {clean.shape}"


        return noisy, clean
