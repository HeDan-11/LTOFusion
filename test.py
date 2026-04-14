import glob
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as ttf

from core.model import ActionNet, PolicyNet


# Please modify the paths below after cloning this repository.

# Dataset root. It supports both SOURCE_DIR/vi + SOURCE_DIR/ir and SOURCE_DIR/MRI-PET/MRI + SOURCE_DIR/MRI-PET/PET.
SOURCE_DIR = "./datasets/MSRS"
# SOURCE_DIR = "./datasets/medical"

# Source modality order. For SOURCE_DIR above, "vi-ir" means SOURCE_DIR/vi and SOURCE_DIR/ir.
# Medical: path1 = MRI  path2 = Other  eg. MODALITY_NAME = "MRI-PET"
MODALITY_NAME = "vi-ir"
# MODALITY_NAME = "MRI-CT"

# Fusion result directory.
SAVE_DIR = f"./results/MSRS/{MODALITY_NAME}"

# Released checkpoint path. The packaged checkpoint is renamed to pth/best.ckpt.
CHECKPOINT_PATH = "./pth/best.ckpt"

# Keep this consistent with the inference step count used by the released model.
# Medical: MAX_STEP = 5
MAX_STEP = 1

# Use 0 by default for easier cross-platform execution.
NUM_WORKERS = 0

# Original implementation uses CUDA inference.
DEVICE = "cuda:0"


class ImagePairDataset(Dataset):
    def __init__(self, data_dir, modality):
        data_dir = Path(data_dir)
        modality_a, modality_b = modality.split("-")
        direct_dir_a = data_dir / modality_a
        direct_dir_b = data_dir / modality_b
        nested_dir_a = data_dir / modality / modality_a
        nested_dir_b = data_dir / modality / modality_b

        if direct_dir_a.is_dir() and direct_dir_b.is_dir():
            image_dir_a, image_dir_b = direct_dir_a, direct_dir_b
        elif nested_dir_a.is_dir() and nested_dir_b.is_dir():
            image_dir_a, image_dir_b = nested_dir_a, nested_dir_b
        else:
            raise FileNotFoundError(
                "Cannot find paired image folders. Tried: "
                f"{direct_dir_a}, {direct_dir_b}, {nested_dir_a}, {nested_dir_b}"
            )

        img_list_a = sorted(glob.glob(str(image_dir_a / "*")))
        img_list_b = sorted(glob.glob(str(image_dir_b / "*")))
        if not img_list_a or not img_list_b:
            raise FileNotFoundError(f"No images found in {image_dir_a} or {image_dir_b}")
        if len(img_list_a) != len(img_list_b):
            raise ValueError(f"Image count mismatch: {image_dir_a} has {len(img_list_a)}, {image_dir_b} has {len(img_list_b)}")

        self.img_list = list(zip(img_list_a, img_list_b))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        path_a, path_b = self.img_list[index]
        img_a = ttf.to_tensor(Image.open(path_a).convert("RGB"))
        img_b = ttf.to_tensor(Image.open(path_b).convert("RGB"))
        return img_a, img_b


def rgb_to_ycbcr(img):
    r, g, b = torch.split(img, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(img):
    y, cb, cr = torch.split(img, 1, dim=1)
    r = y + 1.403 * (cr - 0.5)
    g = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
    b = y + 1.773 * (cb - 0.5)
    return torch.cat([r, g, b], dim=1).clamp(min=0.0, max=1.0)


class FusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = PolicyNet(input_c=3)
        self.action_net = ActionNet()

    def forward(self, state):
        history = self.policy_net(state)
        return self.action_net(history)

    def load_checkpoint(self, checkpoint_path, device):
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.policy_net.load_state_dict(state_dict["policy"], strict=False)
        self.action_net.load_state_dict(state_dict["action"], strict=False)


@torch.no_grad()
def iterative_fusion(model, img_a, img_b, max_step):
    fused = torch.maximum(img_a, img_b)
    for _ in range(max_step):
        state = torch.cat([img_a, img_b, fused], dim=1)
        field = model(state)
        fused = torch.clamp(fused + field, min=0.0, max=1.0)
    return fused


def save_fused_image(fused_y, ycbcr_a, ycbcr_b, modality_name, file_name, save_dir):
    if modality_name in ["vi-ir", "PET-MRI", "SPECT-MRI"]:
        fused_ycbcr = torch.cat([fused_y, ycbcr_a[:, 1:]], dim=1)
    else:
        fused_ycbcr = torch.cat([fused_y, ycbcr_b[:, 1:]], dim=1)

    fused = ycbcr_to_rgb(fused_ycbcr).cpu().permute(2, 3, 1, 0).squeeze().contiguous().numpy()
    fused = np.array(fused * 255, dtype="uint8")
    fused_name = file_name.replace("MRI", "fused")
    Image.fromarray(fused).save(Path(save_dir) / fused_name)


def run_fusion(source_dir, modality_name, checkpoint_path, save_dir, max_step, device):
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset = ImagePairDataset(source_dir, modality_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model = FusionModel().to(device)
    model.load_checkpoint(checkpoint_path, device)
    model.eval()

    for index, (rgb_a, rgb_b) in enumerate(loader):
        file_name = Path(dataset.img_list[index][0]).name
        ycbcr_a = rgb_to_ycbcr(rgb_a.to(device))
        ycbcr_b = rgb_to_ycbcr(rgb_b.to(device))
        fused_y = iterative_fusion(model, ycbcr_a[:, :1], ycbcr_b[:, :1], max_step)
        save_fused_image(fused_y, ycbcr_a, ycbcr_b, modality_name, file_name, save_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Testing %s with checkpoint %s", MODALITY_NAME, CHECKPOINT_PATH)
    run_fusion(SOURCE_DIR, MODALITY_NAME, CHECKPOINT_PATH, SAVE_DIR, MAX_STEP, DEVICE)
    logging.info("Fused images saved to %s", SAVE_DIR)


if __name__ == "__main__":
    main()
