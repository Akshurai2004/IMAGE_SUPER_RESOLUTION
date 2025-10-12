# 🧠 Super-Resolution Image Reconstruction using SRCNN

## 📁 Project Overview

This project implements a **Super-Resolution Convolutional Neural Network (SRCNN)** to enhance low-resolution images into high-resolution outputs.
The model is trained using paired datasets of low- and high-resolution images and can restore image details lost during downsampling.

---

## 📂 Repository Structure

```
├── srcnn1.ipynb                  # Main Jupyter Notebook (training + inference)
├── best_model.pth                # Saved model checkpoint
├── dataset/
│   ├── train/
│   │   ├── low_res/              # Low-resolution training images
│   │   └── high_res/             # High-resolution ground-truth images
│   ├── val/
│   │   ├── low_res/              # Low-resolution validation images
│   │   └── high_res/             # High-resolution validation images
│   ├── raw/                      # Input images for testing/final output
│   └── out/                      # Super-resolved output images
└── README.md                     # Project documentation (this file)
```

---

## ⚙️ Requirements

Make sure you have the following Python libraries installed:

```bash
pip install torch torchvision pillow
```

**Imports used:**

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
```

---

## 🚀 How to Run

You can execute the project in any Python environment that supports Jupyter Notebooks or Python scripts.

### Option 1: Using Jupyter/Colab

1. Open the file **`srcnn1.ipynb`** in Google Colab or VS Code.
2. Run all cells sequentially.
3. Outputs (enhanced images) will be saved under `dataset/out/`.

### Option 2: Using Python Script

If converted to a `.py` file, run it as:

```bash
python3 srcnn1.py
```

---

## 🧩 Model Details

* **Architecture:** SRCNN (5-layer CNN)
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam / SGD
* **Checkpoint:** `best_model.pth` — contains trained weights for inference.

---

## 🧠 Usage

Place your **test images** inside:

```
dataset/raw/
```

After running inference, the **super-resolved images** will appear in:

```
dataset/out/
```

---

## 📊 Results

| Metric | Description                                              |
| ------ | -------------------------------------------------------- |
| PSNR   | Peak Signal-to-Noise Ratio – measures output quality     |
| MSE    | Mean Squared Error between high-res and predicted images |

(Include screenshots or sample output images here if available.)

---

## 🧑‍💻 Author

**Aniruddha K S**
PES University — AIML Section A
SRN: PES1UG23AM905

**Akanksh Rai**
PES University — AIML Section A
SRN: PES1UG23AM031

---

## 📜 License

This project is open for academic and research use.
Feel free to modify and build upon it.

---

