from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB_MODE"] = "disabled"
import json
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from models import get_net
import torch
import torch.optim
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from dataset import DenoisingDataset
import wandb
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio
from util.common_utils import * 
from util.loss import total_variation

wandb.init(project="canlab")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize=-1
dim_div_by = 64

# noisy_dir = '/workspace/TV-DIP/data/test_data/images_combined'
# clean_dir = '/workspace/TV-DIP/data/test_data/images_clean'

# noisy_dir = '/workspace/datasets/noised'
# clean_dir = '/workspace/datasets/png'

noisy_dir = '/workspace/testcode/data/can/noised/combined'
clean_dir = '/workspace/testcode/data/can/png'
# Dataset / Dataloader
dataset = DenoisingDataset(noisy_dir, clean_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

show_every=200
figsize=5
pad = 'reflection' # 'zero'
INPUT = 'noise'
input_depth = 32
OPTIMIZER = 'adam'
OPT_OVER =  'net'
num_iter = 15000
iter_step = 1500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reg_noise_std = 3e-5

net = get_net(
    input_depth=1,                 # IR 입력 1채널
    NET_TYPE='skip',
    pad='reflection',
    upsample_mode='bilinear',
    n_channels=1,                 # 출력 1채널
    skip_n33d=128,
    skip_n33u=128,
    skip_n11=4,
    num_scales=5,
    downsample_mode='stride',
)
model_config = {
    "input_depth": 1,
    "NET_TYPE": "skip",
    "pad": "reflection",
    "upsample_mode": "bilinear",
    "n_channels": 1,
    "skip_n33d": 128,
    "skip_n33u": 128,
    "skip_n11": 4,
    "num_scales": 5,
    "downsample_mode": "stride"
}
timestamp = datetime.now().strftime("%Y%m%d")

save_dir = f'./results/train/{timestamp}/image'
weight_dir = f'./results/train/{timestamp}'
os.makedirs(save_dir, exist_ok=True)

config_path = os.path.join(f'./results/train/{timestamp}', f"model_config_{timestamp}.json")
with open(config_path, "w") as f:
    json.dump(model_config, f, indent=4)

# 장치 설정
net = net.to(device)

# PSNR 추적용
def compute_psnr(tensor_pred, np_gt):
    output_np = tensor_pred.detach().cpu().numpy()
    target_np = np_gt.detach().cpu().numpy()
    
    psnrs = []
    for i in range(output_np.shape[0]):
        pred_img = np.clip(output_np[i, 0], 0, 1)
        target_img = target_np[i, 0]
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)

# Loss & Optimizer
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


# 학습
num_epochs = 5000
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    running_psnr = 0.0
    psnr = 0.0
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        # pred_noise = net(noisy)
        output = net(noisy)
        # tv_loss = total_variation(pred_noise, reduction="mean", step=1)
        # tv_loss = tv_loss.mean()
        # loss = tv_loss 
        # loss = (criterion(pred_noise, clean) + 0.001*tv_loss)
        # denoised = noisy - pred_noise
        loss = criterion(output, clean) #+ 0.001*tv_loss
        psnr = compute_psnr(output,clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_psnr += psnr.item()

    print(f"[Epoch {epoch+1}] Loss: {running_loss / len(loader):.6f}, PSNR {running_psnr / len(loader):.6f}")
    
    ###wandb###
    wandb.log({"Training Loss":running_loss / len(loader)})
    wandb.log({"PSNR" :running_psnr / len(loader)})

    if (epoch + 1) % 500 == 0:
        net.eval()
        with torch.no_grad():
            out_img = output[0].detach().cpu().squeeze().numpy()
            noisy_img = noisy[0].detach().cpu().squeeze().numpy()
            clean_img = clean[0].detach().cpu().squeeze().numpy()

            # 정규화 후 0~255 범위로 저장
            def to_uint8(img):
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return (img_norm * 255).astype(np.uint8)

            imageio.imwrite(os.path.join(save_dir+'', f"epoch{epoch+1}_noisy.png"), to_uint8(noisy_img))
            imageio.imwrite(os.path.join(save_dir, f"epoch{epoch+1}_denoised.png"), to_uint8(out_img))
            imageio.imwrite(os.path.join(save_dir, f"epoch{epoch+1}_clean.png"), to_uint8(clean_img))

    if (epoch+1) % 100 == 0:
        model_path = os.path.join(weight_dir, f"model_full_{timestamp}_{epoch}.pth")
        torch.save(net, model_path)
        print(f"모델이 저장되었습니다: {model_path}")
