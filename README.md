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
# 2. DOWNLOAD AND EXTRACT:
#    - Download CelebA, extract ALL images to data/celeba/img_align_celeba/
#    - Download Face Mask, extract without_mask/ to data/mask/
# 3. CREATE THE DATA FOLDER STRUCTURE:
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

Output: 64x64 RGB faces with smooth latent space interpolation

## Sample Outputs:
https://samples/sample_grid.png
Example generated faces after training


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
