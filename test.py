"""LTOFusion batch inference (use the pt18 environment on Windows).

Examples, from this directory:
  python test.py --tasks vif --vif_datasets MSRS_Seg --dry_run
  python test.py --tasks vif --vif_datasets MSRS M3FD TNO Road --max_images 1
  python test.py --tasks medical
  python test.py --tasks all --max_images 1
  python test.py --tasks medical --save_mode legacy
  python test.py --save_mode legacy95 --method_name LTOFusion_legacy95

Dataset options accept default, all, or explicit names. all scans every dataset
folder and requires valid pairs. --max_images limits each dataset (0 means all).
Existing outputs are skipped; use --overwrite or a different --method_name when
changing weights, iteration counts, or save mode. --dry_run loads no weights and
writes no files. The default checkpoint is resolved relative to this script.

VIF inputs: VI then IR, with color from VI. Medical inputs: MRI then CT/PET/SPECT,
with color from the second input. Default iterations: TNO/Road/M3FD=3,
MSRS/MSRS_Seg and other VIF datasets=1, MRI-CT/MRI-PET/MRI-SPECT=5.
--vif_steps and --medical_steps override their respective dataset defaults.
Inputs are padded to multiples of 8 for the network and cropped before saving.
All save modes keep the original floating-point color restoration and uint8
truncation: cv2 (default) uses OpenCV, legacy uses the original PIL encoder.
legacy95 uses the original PIL pipeline with JPEG quality=95; other formats
are unchanged from legacy.

Outputs retain the input size and extension; MRI in filenames becomes fused.
VIF: <output_root>/<method_name>/<dataset>/ir-vi/<filename>
Medical: <output_root>/<method_name>/<dataset>/<filename>
"""

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

# Import torch first on Windows to avoid image-library DLL conflicts.
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as ttf

from core.model import ActionNet, PolicyNet


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "pth" / "best.ckpt"
DEFAULT_VIF_ROOT = Path(r"E:\python\pytorch\Medical_image_fusion\data\VIF_dataset")
DEFAULT_MEDICAL_ROOT = Path(r"E:\python\pytorch\Medical_image_fusion\data\test_imgs_IN_jpg")
DEFAULT_OUTPUT_ROOT = Path(r"G:\A_Image_fusion_results\A_Image_fusion_results_50")
DEFAULT_VIF_DATASETS = ("MSRS", "M3FD", "TNO", "Road", "MSRS_Seg")
DEFAULT_MEDICAL_DATASETS = ("MRI-CT", "MRI-PET", "MRI-SPECT")
DEFAULT_VIF_STEPS = {
    "msrs": 1,
    "msrs_seg": 1,
    "tno": 3,
    "road": 3,
    "m3fd": 3,
}
DEFAULT_MEDICAL_STEPS = {
    "mri-ct": 5,
    "mri-pet": 5,
    "mri-spect": 5,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Preserve the released LTOFusion VI-first order; source_2/source_1 follows the
# batch reference's convention for anonymously named source folders.
VIF_PAIR_DIRS = (
    ("vi", "ir"), ("visible", "infrared"), ("VIS", "IR"),
    ("source_2", "source_1"), ("source2", "source1"),
)
SAVE_MODE = "cv2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--method_name", default="LTOFusionv3")
    parser.add_argument("--tasks", nargs="+", choices=("all", "vif", "medical"), default=["all"])
    parser.add_argument("--vif_root", type=Path, default=DEFAULT_VIF_ROOT)
    parser.add_argument("--medical_root", type=Path, default=DEFAULT_MEDICAL_ROOT)
    parser.add_argument("--vif_datasets", nargs="+", default=["default"])
    parser.add_argument("--medical_datasets", nargs="+", default=["default"])
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0, or cpu")
    parser.add_argument("--max_images", type=int, default=0, help="Maximum pairs per dataset; 0 means all")
    parser.add_argument("--vif_steps", type=int, default=None,
                        help="Override iterations for all selected VIF datasets (default: TNO/Road/M3FD=3, others=1)")
    parser.add_argument("--medical_steps", type=int, default=None,
                        help="Override iterations for all selected medical datasets (default: MRI-CT/MRI-PET/MRI-SPECT=5)")
    parser.add_argument("--save_mode", choices=("cv2", "legacy", "legacy95"), default=SAVE_MODE,
                        help="cv2: OpenCV (default); legacy: original PIL; legacy95: original PIL with JPEG quality=95")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs; otherwise skip them")
    parser.add_argument("--dry_run", action="store_true", help="Check pairs, sizes and paths without loading weights or writing images")
    opt = parser.parse_args(argv)
    if opt.max_images < 0:
        parser.error("--max_images must be >= 0")
    if ((opt.vif_steps is not None and opt.vif_steps < 1)
            or (opt.medical_steps is not None and opt.medical_steps < 1)):
        parser.error("--vif_steps and --medical_steps must be >= 1")
    if opt.method_name in ("", ".", "..") or any(c in opt.method_name for c in "/\\:"):
        parser.error("--method_name must be a directory name")
    return opt


@dataclass
class PairItem:
    task: str
    dataset: str
    image1: Path
    image2: Path
    output: Path


def natural_key(value):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(value))]


def find_child_dir(parent, name):
    matches = [p for p in parent.iterdir() if p.is_dir() and p.name.lower() == name.lower()]
    if len(matches) > 1:
        raise ValueError("Ambiguous directory %s in %s" % (name, parent))
    return matches[0] if matches else None


def image_files(folder):
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                  key=lambda p: natural_key(p.name))


def normalized_pair_key(path):
    # 只移除明确的模态前后缀；不替换文件名中间的字符，避免破坏 FLIR 等名字。
    stem = path.stem.lower()
    tokens = "infrared|visible|spect|mri|pet|ct|vis|ir|vi"
    stem = re.sub(r"^(?:" + tokens + r")[_\-\s]*", "", stem)
    stem = re.sub(r"[_\-\s]*(?:" + tokens + r")$", "", stem)
    return stem.strip("_ -")


def build_pairs_from_dirs(first, second):
    left, right = image_files(first), image_files(second)
    if not left or not right:
        raise ValueError("Empty image folders: %s / %s" % (first, second))
    # 不按排序 zip；每一步都要求唯一匹配，且每个第二模态文件只能使用一次。
    key_functions = (lambda p: p.name.lower(), lambda p: p.stem.lower(), normalized_pair_key)
    indexes = []
    for key in key_functions:
        index = {}
        for path in right:
            index.setdefault(key(path), []).append(path)
        indexes.append(index)
    used, pairs = set(), []
    for path in left:
        match = None
        for key, index in zip(key_functions, indexes):
            candidates = index.get(key(path), [])
            if len(candidates) > 1:
                raise ValueError("Ambiguous pair for %s: %s" % (path, candidates))
            if candidates:
                match = candidates[0]
                break
        if match is None or match in used:
            raise ValueError("Missing or reused pair for %s in %s" % (path, second))
        used.add(match)
        pairs.append((path, match))
    if len(used) != len(right):
        raise ValueError("Unpaired images in %s: %s" % (second, sorted(set(right) - used)[:5]))
    return pairs


def dataset_names(root, requested, defaults):
    if not root.is_dir():
        raise FileNotFoundError(root)
    if requested == ["default"]:
        return list(defaults)
    if requested == ["all"]:
        return sorted((p.name for p in root.iterdir() if p.is_dir()), key=natural_key)
    if any(name in ("all", "default") for name in requested):
        raise ValueError("Use all/default alone, or provide explicit dataset names")
    for name in requested:
        if name in (".", "..") or any(c in name for c in "/\\:"):
            raise ValueError("Dataset must be a directory name: %s" % name)
    return list(dict.fromkeys(requested))


def pair_dirs(task, folder):
    if task == "vif":
        for first, second in VIF_PAIR_DIRS:
            a, b = find_child_dir(folder, first), find_child_dir(folder, second)
            if a is not None and b is not None:
                return a, b
        raise ValueError("Cannot identify VI/IR folders in %s" % folder)
    if folder.name not in DEFAULT_MEDICAL_DATASETS:
        raise ValueError("Supported medical datasets: %s; got %s" % (DEFAULT_MEDICAL_DATASETS, folder.name))
    a = find_child_dir(folder, "MRI")
    b = find_child_dir(folder, folder.name.split("-", 1)[1])
    if a is None or b is None:
        raise ValueError("Missing medical modality folders in %s" % folder)
    return a, b


def collect_items(opt):
    tasks = ["vif", "medical"] if "all" in opt.tasks else list(dict.fromkeys(opt.tasks))
    items, targets = [], set()
    for task in tasks:
        root = getattr(opt, task + "_root")
        defaults = DEFAULT_VIF_DATASETS if task == "vif" else DEFAULT_MEDICAL_DATASETS
        for name in dataset_names(root, getattr(opt, task + "_datasets"), defaults):
            folder = root / name
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            pairs = build_pairs_from_dirs(*pair_dirs(task, folder))
            selected = pairs[:opt.max_images] if opt.max_images else pairs
            destination = opt.output_root / opt.method_name / name
            if task == "vif":
                destination = destination / "ir-vi"
            print("[%s] %s: pairs=%d selected=%d" % (task, name, len(pairs), len(selected)))
            for a, b in selected:
                with Image.open(a) as im1, Image.open(b) as im2:
                    if im1.size != im2.size:
                        raise ValueError("Image size mismatch: %s %s / %s %s" % (a, im1.size, b, im2.size))
                output = destination / a.name.replace("MRI", "fused")
                key = str(output.resolve()).lower()
                if key in targets:
                    raise ValueError("Output collision: %s" % output)
                targets.add(key)
                items.append(PairItem(task, name, a, b, output))
    if not items:
        raise ValueError("No image pairs selected")
    # 即使指定 overwrite，也不允许把结果写回任意输入图像。
    sources = {str(p.resolve()).lower() for item in items for p in (item.image1, item.image2)}
    if targets & sources:
        raise ValueError("Output paths overlap source images")
    return items


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


def save_fused_image(fused_y, ycbcr_a, ycbcr_b, modality_name, file_name, save_dir, save_mode=SAVE_MODE):
    if save_mode not in ("cv2", "legacy", "legacy95"):
        raise ValueError(f"Unknown save mode: {save_mode}")
    if modality_name in ["vi-ir", "PET-MRI", "SPECT-MRI"]:
        fused_ycbcr = torch.cat([fused_y, ycbcr_a[:, 1:]], dim=1)
    else:
        fused_ycbcr = torch.cat([fused_y, ycbcr_b[:, 1:]], dim=1)

    fused = ycbcr_to_rgb(fused_ycbcr)[0].detach().cpu().permute(1, 2, 0).contiguous().numpy()
    fused = np.array(fused * 255, dtype="uint8")
    fused_name = file_name.replace("MRI", "fused")
    output_path = Path(save_dir) / fused_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_mode == "legacy":
        Image.fromarray(fused).save(output_path)
    elif save_mode == "legacy95":
        save_options = {"quality": 95} if output_path.suffix.lower() in (".jpg", ".jpeg") else {}
        Image.fromarray(fused).save(output_path, **save_options)
    else:
        # OpenCV expects BGR; imencode + tofile also supports Unicode paths.
        fused_bgr = cv2.cvtColor(fused, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(output_path.suffix, fused_bgr)
        if not ok:
            raise RuntimeError(f"Failed to encode image: {output_path}")
        buffer.tofile(str(output_path))


def load_rgb(path, device):
    with Image.open(path) as image:
        return ttf.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)


@torch.no_grad()
def fuse_item(model, item, device, steps, save_mode):
    first = rgb_to_ycbcr(load_rgb(item.image1, device))
    second = rgb_to_ycbcr(load_rgb(item.image2, device))
    height, width = first.shape[-2:]
    padding = (0, (-width) % 8, 0, (-height) % 8)
    y_a, y_b = first[:, :1], second[:, :1]
    if any(padding):
        y_a = F.pad(y_a, padding, mode="replicate")
        y_b = F.pad(y_b, padding, mode="replicate")
    fused_y = iterative_fusion(model, y_a, y_b, steps)[:, :, :height, :width]
    modality = "vi-ir" if item.task == "vif" else item.dataset
    save_fused_image(fused_y, first, second, modality, item.output.name, item.output.parent, save_mode)


def iterations_for_item(opt, item):
    if item.task == "medical":
        if opt.medical_steps is not None:
            return opt.medical_steps
        return DEFAULT_MEDICAL_STEPS.get(item.dataset.lower(), 5)
    if opt.vif_steps is not None:
        return opt.vif_steps
    return DEFAULT_VIF_STEPS.get(item.dataset.lower(), 1)


def run_batch(opt):
    vif_steps = opt.vif_steps if opt.vif_steps is not None else DEFAULT_VIF_STEPS
    medical_steps = opt.medical_steps if opt.medical_steps is not None else DEFAULT_MEDICAL_STEPS
    print("Save mode: %s | iterations: vif=%s medical=%s" %
          (opt.save_mode, vif_steps, medical_steps), flush=True)
    items = collect_items(opt)
    if opt.dry_run:
        for item in items:
            steps = iterations_for_item(opt, item)
            print("[DRY][steps=%d] %s + %s -> %s" % (steps, item.image1, item.image2, item.output))
        print("Checked %d pairs; no files written." % len(items))
        return
    pending = []
    for item in items:
        if item.output.exists() and not item.output.is_file():
            raise ValueError("Output is not a file: %s" % item.output)
        if opt.overwrite or not item.output.exists():
            pending.append(item)
    skipped = len(items) - len(pending)
    print("Selected=%d pending=%d skipped=%d (existing files are not revalidated)" %
          (len(items), len(pending), skipped), flush=True)
    if not pending:
        return
    device = torch.device(opt.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use --device cpu or a CUDA environment")
        torch.cuda.set_device(device)
    model = FusionModel().to(device)
    model.load_checkpoint(str(opt.checkpoint_path), device)
    model.eval()
    print("Weight: %s" % opt.checkpoint_path, flush=True)
    start = time.perf_counter()
    stats = {}
    for index, item in enumerate(pending, 1):
        before = time.perf_counter()
        steps = iterations_for_item(opt, item)
        print("[%d/%d] %s/%s steps=%d" %
              (index, len(pending), item.dataset, item.image1.name, steps), flush=True)
        fuse_item(model, item, device, steps, opt.save_mode)
        count, seconds = stats.get((item.task, item.dataset), (0, 0.0))
        stats[(item.task, item.dataset)] = (count + 1, seconds + time.perf_counter() - before)
    for (task, name), (count, seconds) in stats.items():
        print("[DONE] %s/%s images=%d time=%.2fs" % (task, name, count, seconds))
    print("Saved=%d skipped=%d total=%.2fs" % (len(pending), skipped, time.perf_counter() - start))


def main():
    run_batch(parse_args())


if __name__ == "__main__":
    main()
