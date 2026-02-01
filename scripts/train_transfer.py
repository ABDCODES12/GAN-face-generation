#!/usr/bin/env python3
"""
Transfer learning from CelebA to mask dataset
Original complete version
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image

# Set random seed for reproducibility
torch.manual_seed(42)

# ==================== CORRECT PATHS ====================
# USER MUST UPDATE THESE PATHS
MASK_DATA_DIR = "C:/Users/yourusername/data/mask/without_mask/"  # UPDATE THIS!
G_PATH = "G_final.pth"  # From CelebA training
D_PATH = "D_final.pth"  # From CelebA training

# ==================== CUSTOM DATASET FOR FLAT FOLDER ====================
class FlatImageDataset(Dataset):
    """Custom dataset for images in a flat folder (no subdirectories)"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for file in os.listdir(root_dir):
            if file.lower().endswith(valid_extensions):
                self.image_files.append(file)
        
        print(f"✓ Found {len(self.image_files)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Return image with dummy label (0) since we don't need labels
            return image, 0
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 64, 64), 0

# ==================== HYPERPARAMETERS ====================
image_size = 64
batch_size = 128
latent_size = 256
epochs = 40        
lr_d = 0.0001     
lr_g = 0.0001
beta1 = 0.5
beta2 = 0.999

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# ==================== DATA LOADING ====================
print("="*80)
print("TRANSFER LEARNING - MASKED FACE GENERATION")
print("="*80)
print(f"Loading dataset from: {MASK_DATA_DIR}")

# Check if directory exists
if not os.path.exists(MASK_DATA_DIR):
    print(f"\n❌ ERROR: Directory not found: {MASK_DATA_DIR}")
    print("\nTo run transfer learning:")
    print("1. Download Face Mask dataset from: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset")
    print("2. Extract 'without_mask' folder to the path above")
    print("3. Update MASK_DATA_DIR in this script with your actual path")
    print("="*80)
    exit()

train_ds = FlatImageDataset(
    root_dir=MASK_DATA_DIR,
    transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)
    ])
)

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Helper Functions
def denorm(img_tensors):
    """Denormalize image tensors"""
    if isinstance(img_tensors, torch.Tensor):
        return img_tensors * stats[1][0] + stats[0][0]
    return img_tensors

def show_images(images, nmax=64):
    """Display images"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    
    images_cpu = images.detach().cpu()[:nmax]
    grid = make_grid(denorm(images_cpu), nrow=8).permute(1, 2, 0)
    grid = torch.clamp(grid, 0, 1)
    ax.imshow(grid.numpy())
    plt.show()

def show_batch(dl, nmax=64):
    """Display a batch of images"""
    for images, _ in dl:
        show_images(images, nmax)
        break

# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

train_dl_device = DeviceDataLoader(train_dl, device)

# Your D_final.pth has spectral norm, so we need to match it
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Spectral normalization wrapper
        def add_spectral_norm(module):
            if isinstance(module, nn.Conv2d):
                return nn.utils.spectral_norm(module)
            return module
        
        self.model = nn.Sequential(
            # Input: 3 x 64 x 64
            add_spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            add_spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            add_spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            add_spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 1 x 1 x 1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.model(x)

# ==================== GENERATOR ====================
class Generator(nn.Module):
    def __init__(self, latent_size=256):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# ==================== LOAD PRE-TRAINED MODELS ====================
print("\n" + "="*60)
print("LOADING PRE-TRAINED MODELS")
print("="*60)

# Initialize models
discriminator = Discriminator()
generator = Generator(latent_size)

# Move to device
discriminator = discriminator.to(device)
generator = generator.to(device)

# Load function with spectral norm support
def load_pretrained_models():
    """Load pre-trained models from given paths"""
    print(f"Generator path: {G_PATH}")
    print(f"Discriminator path: {D_PATH}")
    
    # Check if files exist
    if not os.path.exists(G_PATH):
        print("❌ G_final.pth not found at the specified path!")
        print("You must train on CelebA first: python train_celeba.py")
        return False
    if not os.path.exists(D_PATH):
        print("❌ D_final.pth not found at the specified path!")
        print("You must train on CelebA first: python train_celeba.py")
        return False
    
    try:
        generator.load_state_dict(torch.load(G_PATH, map_location=device))
        
        state_dict = torch.load(D_PATH, map_location=device)
        
        # Create new state dict without spectral norm suffix
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove spectral norm suffixes if present
            if key.endswith('_orig'):
                new_key = key.replace('_orig', '')
                new_state_dict[new_key] = value
            elif not (key.endswith('_u') or key.endswith('_v')):
                # Keep other keys (except spectral norm vectors)
                new_state_dict[key] = value
        
        # Load with strict=False to ignore missing keys
        discriminator.load_state_dict(new_state_dict, strict=False)        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Starting from scratch...")
        return False

# Load models
transfer_success = load_pretrained_models()

if not transfer_success:
    print("\n❌ Failed to load pre-trained models.")
    print("Make sure you have:")
    print("1. Trained on CelebA first (run train_celeba.py)")
    print("2. Have G_final.pth and D_final.pth in the same directory")
    exit()

# Print model info
print("\n" + "="*50)
print("MODEL INFORMATION")
print("="*50)
print(f"Dataset size: {len(train_ds):,} images")
print(f"Batch size: {batch_size}")
print("="*50)

# ==================== TRAINING FUNCTIONS ====================
def train_discriminator(real_images, opt_d):
    """Train the discriminator"""
    opt_d.zero_grad()
    
    batch_size = real_images.size(0)
    
    # Create labels for real and fake images
    real_labels = torch.ones(batch_size, 1, device=device)  # Real = 1
    fake_labels = torch.zeros(batch_size, 1, device=device)  # Fake = 0
    
    # Train with real images
    real_preds = discriminator(real_images)
    real_loss = F.binary_cross_entropy(real_preds, real_labels)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Train with fake images
    fake_preds = discriminator(fake_images.detach())
    fake_loss = F.binary_cross_entropy(fake_preds, fake_labels)
    fake_score = torch.mean(fake_preds).item()
    
    # Total discriminator loss
    loss_d = real_loss + fake_loss
    
    # Backpropagation
    loss_d.backward()
    opt_d.step()
    
    return loss_d.item(), real_score, fake_score

def train_generator(opt_g):
    """Train the generator"""
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator (make it output 1 for fakes)
    fake_preds = discriminator(fake_images)
    fake_labels = torch.ones(batch_size, 1, device=device)  # We want discriminator to say "real"
    
    loss_g = F.binary_cross_entropy(fake_preds, fake_labels)
    
    # Backpropagation
    loss_g.backward()
    opt_g.step()
    
    return loss_g.item()

# ==================== SAMPLE SAVING ====================
sample_dir = 'generated_transfer'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    """Save generated samples"""
    with torch.no_grad():
        fake_images = generator(latent_tensors)
        fake_fname = f'transfer-generated-{index:04d}.png'
        
        # Save to file
        save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
        
        if show:
            # Display the images
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            
            # Move to CPU for display
            images_cpu = fake_images.cpu()
            grid = make_grid(denorm(images_cpu), nrow=8).permute(1, 2, 0)
            grid = torch.clamp(grid, 0, 1)
            ax.imshow(grid.numpy())
            plt.title(f'Transfer Learning - Epoch {index}')
            plt.show()
    
    print(f"Saved {fake_fname}")

# ==================== TRAINING LOOP ====================
def fit(epochs, lr_d, lr_g, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    
    # Fixed latent for consistent sample generation
    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    
    for epoch in range(epochs):
        epoch_loss_g = 0
        epoch_loss_d = 0
        epoch_real_score = 0
        epoch_fake_score = 0
        num_batches = 0
        
        pbar = tqdm(train_dl_device, desc=f'Epoch {epoch+1}/{epochs}')
        for real_images, _ in pbar:
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            
            # Train generator
            loss_g = train_generator(opt_g)
            
            # Accumulate metrics
            epoch_loss_g += loss_g
            epoch_loss_d += loss_d
            epoch_real_score += real_score
            epoch_fake_score += fake_score
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss_g': f'{loss_g:.3f}',
                'loss_d': f'{loss_d:.3f}',
                'D(real)': f'{real_score:.3f}',
                'D(fake)': f'{fake_score:.3f}'
            })
        
        # Average metrics
        avg_loss_g = epoch_loss_g / max(num_batches, 1)
        avg_loss_d = epoch_loss_d / max(num_batches, 1)
        avg_real_score = epoch_real_score / max(num_batches, 1)
        avg_fake_score = epoch_fake_score / max(num_batches, 1)
        
        # Record losses & scores
        losses_g.append(avg_loss_g)
        losses_d.append(avg_loss_d)
        real_scores.append(avg_real_score)
        fake_scores.append(avg_fake_score)
        
        # Log metrics
        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"loss_g: {avg_loss_g:.4f}, loss_d: {avg_loss_d:.4f}, "
              f"real_score: {avg_real_score:.4f}, fake_score: {avg_fake_score:.4f}")
        
        # Save samples every epoch
        save_samples(epoch + start_idx, fixed_latent, show=(epoch % 10 == 0))
        
        # Save checkpoints periodically
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'G_transfer_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'D_transfer_epoch_{epoch+1}.pth')
            print(f"Transfer checkpoint saved at epoch {epoch+1}")
    
    return losses_g, losses_d, real_scores, fake_scores

# ==================== TRAINING ====================
print("\n" + "="*60)
print("STARTING TRANSFER LEARNING TRAINING")
print("="*60)
print(f"Mask Dataset: {len(train_ds):,} images")
print(f"Pre-trained models: {'✅ LOADED' if transfer_success else '❌ NOT FOUND'}")
print(f"Fine-tuning for: {epochs} epochs")
print("="*60)

# Show sample batch from dataset
print("\nSample batch from mask dataset:")
show_batch(train_dl)

# Generate INITIAL samples from pre-trained model
print("\nInitial samples (from pre-trained CelebA model):")
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(0, fixed_latent, show=True)

# Start training
print("\nBeginning fine-tuning on mask dataset...")
history = fit(epochs, lr_d, lr_g, start_idx=1)

# Save final models
torch.save(generator.state_dict(), 'G_transfer_final.pth')
torch.save(discriminator.state_dict(), 'D_transfer_final.pth')

# ==================== VISUALIZATION ====================
losses_g, losses_d, real_scores, fake_scores = history

# Plot losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses_d, '-', linewidth=2, label='Discriminator')
plt.plot(losses_g, '-', linewidth=2, label='Generator')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Transfer Learning Losses')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(real_scores, '-', linewidth=2, label='Real Score')
plt.plot(fake_scores, '-', linewidth=2, label='Fake Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Discriminator Scores (Transfer)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_training_metrics.png', dpi=100, bbox_inches='tight')
plt.show()

# ==================== GENERATE FINAL SAMPLES ====================
print("\nGenerating final transfer learning samples...")
with torch.no_grad():
    for i in range(5):
        random_latent = torch.randn(64, latent_size, 1, 1, device=device)
        save_samples(1000 + i, random_latent, show=(i == 0))

print("\n" + "="*60)
print("TRANSFER LEARNING COMPLETE!")
print("="*60)
print(f"• Original CelebA: 60 epochs, 100K images")
print(f"• Transfer Learning: {epochs} epochs, {len(train_ds):,} HD mask images")
print(f"• Time saved: {60-epochs} fewer epochs")
print("="*60)