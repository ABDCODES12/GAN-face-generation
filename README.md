# 🎭 Face Generation GAN with Transfer Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A complete PyTorch implementation of DCGAN for celebrity face generation, adapted via transfer learning to generate masked faces.

## ✨ Features

- 🏗️ **DCGAN Implementation**: From-scratch implementation with spectral normalization
- 🔄 **Transfer Learning**: Fine-tune pre-trained models on new datasets (33% faster convergence)
- 📊 **Training Visualization**: Real-time metrics tracking and sample generation
- 🎯 **Production Ready**: Modular design, configuration management, logging
- 🖼️ **High-Quality Output**: 64x64 RGB face generation
- 🧪 **Test Mode**: Run with synthetic data - no downloads needed!

## ⚠️ Dataset Requirements

**This project WILL NOT WORK until you download the required datasets.**

### **Required Datasets:**

| Dataset | Size | Download Instructions |
|---------|------|---------------------|
| **CelebA** | ~1.4 GB | [Kaggle Link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)<br>Download `img_align_celeba.zip` → Extract to `data/celeba/img_align_celeba/` |
| **Face Mask** | ~200 MB | [Kaggle Link](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)<br>Extract `without_mask` folder to `data/mask/` |


## 📊 Full Training (With Datasets)
# 1. Download datasets (see instructions above)
# 2. Download & Extract:
    - Download CelebA, extract ALL images to data/celeba/img_align_celeba/
    - Download Face Mask, extract without_mask/ to data/mask/
# 3. Create the data folder structure:
```bash
mkdir -p data/celeba/img_align_celeba
mkdir -p data/mask/without_mask
```
# 4. Update config files with your paths
```bash
#    Edit: configs/celeba_config.yaml
#    Edit: configs/transfer_config.yaml
```

# 5. Install dependencies
```bash
pip install -r requirements.txt
```

# 6. Run Phase 1: Train on CelebA
```bash
python scripts/train_celeba.py
```

# 7. Run Phase 2: Transfer learning
```bash
python scripts/train_transfer.py
```

## ⚙️ Configuration
Update these paths in the config files:
configs/celeba_config.yaml:
```bash
dataset:
  path: "data/celeba/img_align_celeba/"  # ← UPDATE THIS
```
configs/transfer_config.yaml:
```bash
dataset:
  path: "data/mask/without_mask/"  # ← UPDATE THIS
```
## 📈 Results
Training Metrics:
CelebA: 60 epochs, 200K+ images, realistic face generation
Transfer Learning: 40 epochs, 5K images, 33% faster convergence

# Training Progress Visualization

## CelebA Dataset Training

| Epoch 1 | Epoch 5 | Epoch 40 | Epoch 60 (Final) |
|---------|---------|----------|------------------|
| ![Epoch 1](https://github.com/user-attachments/assets/8cb1fe7f-648d-470e-80ee-c12a5cabb6fa) | ![Epoch 5](https://github.com/user-attachments/assets/8760c882-d978-4428-b3ef-1442598365a7) | ![Epoch 40](https://github.com/user-attachments/assets/7e75b676-b83a-4e0f-bedf-549172cd2249) | ![Epoch 60](https://github.com/user-attachments/assets/38e305ed-6529-4dcd-abbf-7ee0338b4e5d) |

## Transfer Learning on FakeFaces Dataset

| After 5 Epochs (Epoch 65) | After 10 Epochs (Epoch 70) |
|---------------------------|----------------------------|
| ![Epoch 65](https://github.com/user-attachments/assets/3ef4f791-c526-4be1-8db6-ab8418749b3e) | ![Epoch 70](https://github.com/user-attachments/assets/91cd27af-2224-43e4-b52b-33a69801ac24) |

### Observations:
1. **Epochs 1-40**: Gradual improvement in face structure and details
2. **Epoch 60**: Best results on CelebA dataset
3. **Transfer Learning**: Rapid adaptation to FakeFaces dataset showing noticeable improvements






📄 License
Distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgments
CelebA Dataset

Face Mask Dataset

PyTorch Team

DCGAN Paper Authors

📧 Contact
Abdullah Hammam - abdullahhammam006@gmail.com

Project Link: https://github.com/ABDCODES12/GAN-face-generation.git
