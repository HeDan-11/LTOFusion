# LTOFusion
LTOFusion: A Learning-to-Optimize Framework With Flow Matching for Unsupervised Image Fusion. This paper has been accepted by IEEE Transactions on Image Processing (TIP) in February 2026.

Research Focus: Infrared-visible light image fusion and medical image fusion.

## Dataset Layout

Before running, prepare one dataset with two modality folders. For the default `MODALITY_NAME = "vi-ir"`, the expected layout is:

```text
MSRS/
  vi/
    0001.png
  ir/
    0001.png
```

For medical data, this layout is also supported:

```text
medical/
  MRI-PET/
    MRI/
      01.jpg
    PET/
      01.jpg
```

## Run

Open `test.py` and modify the marked configuration block:

```python
SOURCE_DIR = "./datasets/MSRS"
MODALITY_NAME = "vi-ir"
SAVE_DIR = f"./results/MSRS/{MODALITY_NAME}"
CHECKPOINT_PATH = "./pth/best.ckpt"
MAX_STEP = 1
DEVICE = "cuda:0"
```

Then run:

```bash
python test.py
```

The fused images will be saved under `SAVE_DIR`.

**Please note that since the research team is still conducting further research based on this code, the training code will not be made public at this time. We appreciate your understanding.**


## Citation
If this code is helpful, please cite the corresponding paper.:

```bibtex
@ARTICLE{11433513_LTOFusion,
  author={He, Dan and Yang, Lijian and Wang, Guofen and Huang, Yuping and Shu, Yucheng and Li, Weisheng},
  journal={IEEE Transactions on Image Processing}, 
  title={LTOFusion: A Learning-to-Optimize Framework With Flow Matching for Unsupervised Image Fusion}, 
  year={2026},
  volume={35},
  number={},
  pages={2857-2872},
  doi={10.1109/TIP.2026.3671658}}
```

## Contact
If you encounter issues or wish to report bugs, please open a GitHub Issue in this repository or contact the maintainers listed on the project page.
