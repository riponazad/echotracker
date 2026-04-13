# EchoTracker: Advancing Myocardial Point Tracking in Echocardiography

> Official code repository &nbsp;·&nbsp; Accepted at **MICCAI 2024** (top 11%)

If you find this work useful, please consider giving the repository a ⭐ — it helps others discover the project and motivates further development!

<p align="center">
  <a href="https://link.springer.com/chapter/10.1007/978-3-031-72083-3_60">
    <img src="https://img.shields.io/badge/MICCAI_2024-Paper-4b8bbe?style=flat-square&logo=springer&logoColor=white" alt="Paper">
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2405.08587">
    <img src="https://img.shields.io/badge/arXiv-2405.08587-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  &nbsp;
  <a href="https://riponazad.github.io/echotracker/">
    <img src="https://img.shields.io/badge/Project-Page-2ea44f?style=flat-square" alt="Project Page">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/riponazad/echotracker">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-ff9f00?style=flat-square" alt="HF Demo">
  </a>
  &nbsp;
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT%20%2F%20Apache%202.0-lightgrey?style=flat-square" alt="License">
  </a>
</p>

<p align="center">
  <img src="https://github.com/riponazad/echotracker/blob/main/assets/UltraTRacking.png" width="800">
</p>

---

## Updates

- **(13.04.26)** A web-based demo is now live on Hugging Face Spaces — try EchoTracker instantly in your browser with no local setup: [🤗 HF Demo](https://huggingface.co/spaces/riponazad/echotracker).
- **(13.04.26)** `demo2` is redesigned as a fully interactive single-window GUI — play the video, click to select points, run tracking, and view EchoTracker vs TAPIR results side-by-side, all without leaving the window.
- **(19.11.24)** Fine-tuning and inference code for TAPIR added. Licensing updated accordingly.
- **(19.11.24)** The interactive demo (`demo2`) now visualizes tracking results from both EchoTracker and TAPIR together.

---

## Table of Contents

- [Installation](#installation)
- [Model Checkpoints](#model-checkpoints)
- [Run Demos](#run-demos)
- [Train / Evaluation](#trainevaluation)
- [Citation](#citation)
- [License](#license)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/riponazad/echotracker.git
cd echotracker

# 2. Create and activate a conda environment
conda create --name echotracker python=3.11
conda activate echotracker

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the project root to PYTHONPATH
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```

---

## Model Checkpoints

Download each checkpoint and place the file in the corresponding directory before running any demo.

| Model | Checkpoint | Destination |
|---|---|---|
| EchoTracker | [`model-000000000.pth`](https://drive.google.com/file/d/1FJmNlfE5lNPSpBbtnc3KBOgC7cp6DCpl/view?usp=drive_link) | `model/weights/echotracker/` |
| TAPIR (fine-tuned) | [`model-000000099.pth`](https://drive.google.com/file/d/1mkj_8MQo0ntupAt3IBZ6XtjXUuDKMMDc/view?usp=drive_link) | `model/weights/tapir/finetuned/` |

> **Note on the Hugging Face Demo weights:** The EchoTracker model served in the [🤗 HF Demo](https://huggingface.co/spaces/riponazad/echotracker) uses an **updated checkpoint** from a follow-up work:
> [*Taming Modern Point Tracking for Speckle Tracking Echocardiography via Impartial Motion*](https://ieeexplore.ieee.org/document/11374398) — ICCV 2025 Workshop &nbsp;·&nbsp; [[arXiv](https://arxiv.org/abs/2507.10127)]
>
> This version achieves best performance when query points are selected from the frame located at approximately **72% of the video's time dimension** — corresponding to **diastasis** (the quiescent slow-filling phase between the E-wave and A-wave) in a full ED-to-ED cardiac cycle (e.g. frame 72 of a 100-frame sequence).
> If you use this model, please also cite that paper — see the [Citation](#citation) section.

---

## Run Demos

### Demo 1 — Quantitative evaluation

```bash
python demo1.py
```

Loads EchoTracker and the bundled ultrasound sample data, estimates trajectories for the preset query points, and prints performance metrics.

---

### Demo 2 — Interactive single-window GUI

```bash
python demo2.py <video_name>
```

> `<video_name>` is the filename **without** the `.mp4` extension. The script looks for the video in the `data/` folder, so the full path resolved is `data/<video_name>.mp4`.
> To use your own video, copy it into the `data/` folder first and pass its name accordingly.
>
> Example: `python demo2.py input` → loads `data/input.mp4`

Everything happens inside one matplotlib window — no separate video players or pop-up figures.

| Step | Action |
|---|---|
| **1. Watch** | The video plays automatically once models are loaded. |
| **2. Pause** | Click **"Pause & Select Points"** to freeze on the first frame. |
| **3. Select** | **Left-click** anywhere on the frame to place tracking points (up to 40). |
| **4. Clear** | Click **"Clear Points"** to remove all placed points and start over. |
| **5. Track** | Click **"Run Tracking"** — both EchoTracker and TAPIR run in the background. |
| **6. Review** | Results appear **side-by-side** with preserved aspect ratio (EchoTracker left, TAPIR right). The combined video is saved to `results/output.mp4`. |
| **7. Repeat** | Click **"Pause & Select Points"** again at any time to try a different set of points. |

<p align="center">
  <img src="https://github.com/riponazad/echotracker/blob/main/assets/output.gif" width="700">
</p>

> **No local setup?** Try the model directly in your browser via the [🤗 Hugging Face Demo](https://huggingface.co/spaces/riponazad/echotracker).

---

## Train/Evaluation

> The training datasets are not publicly available. The code below shows how to plug in your own data.

### EchoTracker

Training and inference are exposed as simple class methods in [`model/net.py`](model/net.py).

```python
from model.net import EchoTracker

B = 1   # batch size
S = 24  # sequence length

# Provide your own data loader
dataloaders, dataset_size = load_datasets(B=B, S=S, use_aug=True)

model = EchoTracker(device_ids=[0])
# model.load(path="model/weights/echotracker", eval=False)  # uncomment to fine-tune
model.load(eval=False)
model.train(
    dataloaders=dataloaders,
    dataset_size=dataset_size,
    log_dir="logs/echotracker",
    ckpt_path="model/weights/echotracker",
    epochs=100,
)
```

### TAPIR (fine-tuning)

```python
from model.net import TAPIR

B = 1   # batch size
S = 24  # sequence length

# Provide your own data loader
dataloaders, dataset_size = load_datasets(B=B, S=S, use_aug=True)

model = TAPIR(pyramid_level=0)
# model.load(path="model/weights/tapir/finetuned", eval=False)  # uncomment to continue fine-tuning
model.load(eval=False)
model.finetune(
    dataloaders=dataloaders,
    dataset_size=dataset_size,
    log_dir="logs/tapir",
    ckpt_path="model/weights/tapir/finetuned",
    epochs=100,
)
```

---

## Citation

If you use this code or the EchoTracker model (MICCAI 2024), please cite:

```bibtex
@InProceedings{azad2024echo,
  author    = {Azad, Md Abulkalam and Chernyshov, Artem and Nyberg, John
               and Tveten, Ingrid and Lovstakken, Lasse and Dalen, H{\aa}vard
               and Grenne, Bj{\o}rnar and {\O}stvik, Andreas},
  title     = {EchoTracker: Advancing Myocardial Point Tracking in Echocardiography},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
  year      = {2024},
  volume    = {IV},
  publisher = {Springer Nature Switzerland},
  pages     = {645--655},
}
```

If you use the updated model weights available in the [🤗 Hugging Face Demo](https://huggingface.co/spaces/riponazad/echotracker), please additionally cite:

```bibtex
@InProceedings{Azad_2025_ICCV,
  author    = {Azad, Md Abulkalam and Nyberg, John and Dalen, H{\aa}vard
               and Grenne, Bj{\o}rnar and Lovstakken, Lasse and {\O}stvik, Andreas},
  title     = {Taming Modern Point Tracking for Speckle Tracking Echocardiography via Impartial Motion},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  month     = {October},
  year      = {2025},
  pages     = {1115--1124},
}
```

---

## License

This project uses a dual-license structure due to third-party code inclusion.

| Component | License |
|---|---|
| EchoTracker (original code) | [MIT License](LICENSE) |
| TAPIR integration ([source](https://github.com/google-deepmind/tapnet)) | [Apache License 2.0](LICENSE) |

When using this project, retain the respective license notices for each component. See the `LICENSE` file for full terms.
