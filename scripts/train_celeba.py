#!/usr/bin/env python3
"""
CelebA Face Generation with GANs - Training from scratch
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

# ==================== CELEBA DATASET PATH ====================
# USER MUST UPDATE THIS PATH
CELEBA_DATA_DIR = "C:/Users/yourusername/data/celeba/img_align_celeba/"  # UPDATE THIS!

# ==================== CUSTOM DATASET FOR FLAT FOLDER ====================
class FlatImageDataset(Dataset):
    """Custom dataset for images in a flat folder (no subdirectories)"""
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for file in sorted(os.listdir(root_dir)):  # Sort for consistency
            if file.lower().endswith(valid_extensions):
                self.image_files.append(file)
        
        # Limit dataset size if specified (useful for testing)
        if limit is not None and limit < len(self.image_files):
            self.image_files = self.image_files[:limit]
        
        print(f"✓ Found {len(self.image_files)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Return image with dummy label (0) since we don't need labels for GANs
            return image, 0
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder image as fallback
            return torch.zeros(3, 64, 64), 0

# ==================== HYPERPARAMETERS ====================
image_size = 64
batch_size = 128
latent_size = 256
epochs = 60                    # Train for 60 epochs (standard for CelebA)
lr_d = 0.0002                 # Learning rate for discriminator
lr_g = 0.0002                 # Learning rate for generator
beta1 = 0.5
beta2 = 0.999

# Normalization stats for CelebA (mean=0.5, std=0.5 for [-1, 1] range)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# ==================== DATA LOADING - CELEBA ====================
print("="*80)
print("CELEBA FACE GENERATION GAN - TRAINING FROM SCRATCH")
print("="*80)
print(f"Loading CelebA dataset from: {CELEBA_DATA_DIR}")

# Check if directory exists
if not os.path.exists(CELEBA_DATA_DIR):
    print(f"\n❌ ERROR: Directory not found: {CELEBA_DATA_DIR}")
    print("\nTo run this project:")
    print("1. Download CelebA from: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
    print("2. Extract to: C:/Users/yourusername/data/celeba/img_align_celeba/")
    print("3. Update CELEBA_DATA_DIR in this script with your actual path")
    print("="*80)
    exit()

print(f"✓ Using directory: {CELEBA_DATA_DIR}")

# Create transforms for CelebA
celeba_transforms = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
])

# Load CelebA dataset using custom dataset class
train_ds = FlatImageDataset(
    root_dir=CELEBA_DATA_DIR,
    transform=celeba_transforms,
    limit=None  # Set to 10000 for faster testing if needed
)

# Create data loader
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

# ==================== HELPER FUNCTIONS ====================
def denorm(img_tensors):
    """Denormalize image tensors from [-1, 1] to [0, 1]"""
    if isinstance(img_tensors, torch.Tensor):
        return img_tensors * stats[1][0] + stats[0][0]
    return img_tensors

def show_images(images, nmax=64, title="Generated Images"):
    """Display images in a grid"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    
    images_cpu = images.detach().cpu()[:nmax]
    grid = make_grid(denorm(images_cpu), nrow=8).permute(1, 2, 0)
    grid = torch.clamp(grid, 0, 1)
    ax.imshow(grid.numpy())
    plt.title(title)
    plt.show()

def show_batch(dl, nmax=64):
    """Display a batch of images from dataloader"""
    for images, _ in dl:
        show_images(images, nmax, title="CelebA Training Images")
        break

# ==================== DEVICE SETUP ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

def to_device(data, device):
    """Move data to specified device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrapper to move batches to device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

train_dl_device = DeviceDataLoader(train_dl, device)

# ==================== DISCRIMINATOR ====================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
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

# ==================== WEIGHT INITIALIZATION ====================
def weights_init(m):
    """Initialize weights for DCGAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ==================== INITIALIZE MODELS ====================
print("\n" + "="*60)
print("INITIALIZING MODELS FROM SCRATCH")
print("="*60)

discriminator = Discriminator()
generator = Generator(latent_size)

# Apply weight initialization
discriminator.apply(weights_init)
generator.apply(weights_init)

# Move to device
discriminator = discriminator.to(device)
generator = generator.to(device)

# Print model info
print("\n" + "="*50)
print("MODEL INFORMATION")
print("="*50)
print(f"Dataset: CelebA")
print(f"Dataset size: {len(train_ds):,} images")
print(f"Batch size: {batch_size}")
print(f"Image size: {image_size}x{image_size}")
print(f"Latent size: {latent_size}")
print("="*50)

# ==================== TRAINING FUNCTIONS ====================
def train_discriminator(real_images, opt_d):
    """Train the discriminator on real and fake images"""
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
    """Train the generator to fool the discriminator"""
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
sample_dir = 'generated_celeba'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    """Save and display generated samples"""
    with torch.no_grad():
        fake_images = generator(latent_tensors)
        fake_fname = f'celeba-generated-{index:04d}.png'
        
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
            plt.title(f'CelebA Generated - Epoch {index}')
            plt.show()
    
    print(f"✓ Saved {fake_fname}")

# ==================== TRAINING LOOP ====================
def fit(epochs, lr_d, lr_g, start_idx=1):
    """Main training loop"""
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    
    # Learning rate schedulers for better convergence
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-6)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)
    
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
        
        # Update learning rates
        scheduler_d.step()
        scheduler_g.step()
        
        # Log metrics
        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"loss_g: {avg_loss_g:.4f}, loss_d: {avg_loss_d:.4f}, "
              f"real_score: {avg_real_score:.4f}, fake_score: {avg_fake_score:.4f}")
        
        # Save samples every epoch
        save_samples(epoch + start_idx, fixed_latent, show=(epoch % 10 == 0))
        
        # Save checkpoints periodically
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'G_celeba_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'D_celeba_epoch_{epoch+1}.pth')
            print(f"✓ Checkpoint saved at epoch {epoch+1}")
    
    return losses_g, losses_d, real_scores, fake_scores

# ==================== START TRAINING ====================
print("\n" + "="*60)
print("STARTING CELEBA GAN TRAINING FROM SCRATCH")
print("="*60)
print(f"Dataset: CelebA ({len(train_ds):,} images)")
print(f"Training epochs: {epochs}")
print(f"Batch size: {batch_size}")
print("="*60)

# Show sample batch from dataset
print("\nSample batch from CelebA dataset:")
show_batch(train_dl)

# Generate and show initial samples (random noise)
print("\nInitial generated samples (before training - random noise):")
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(0, fixed_latent, show=True)

# Start training
print("\n" + "="*60)
print("BEGINNING TRAINING...")
print("="*60)

history = fit(epochs, lr_d, lr_g, start_idx=1)

# Save final models
torch.save(generator.state_dict(), 'G_final.pth')
torch.save(discriminator.state_dict(), 'D_final.pth')
print("\n✓ Final models saved as G_final.pth and D_final.pth")

# ==================== VISUALIZATION ====================
losses_g, losses_d, real_scores, fake_scores = history

# Plot losses and scores
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(losses_d, '-', linewidth=2, label='Discriminator Loss')
plt.plot(losses_g, '-', linewidth=2, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('CelebA GAN Training Losses')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(real_scores, '-', linewidth=2, label='D(real) Score')
plt.plot(fake_scores, '-', linewidth=2, label='D(fake) Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Discriminator Scores on CelebA')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('celeba_training_metrics.png', dpi=100, bbox_inches='tight')
plt.show()

# ==================== GENERATE FINAL SAMPLES ====================
print("\n" + "="*60)
print("GENERATING FINAL SAMPLES")
print("="*60)

with torch.no_grad():
    # Generate multiple batches of samples
    for i in range(5):
        random_latent = torch.randn(64, latent_size, 1, 1, device=device)
        save_samples(1000 + i, random_latent, show=(i == 0))

# ==================== LATENT SPACE INTERPOLATION ====================
print("\n" + "="*60)
print("LATENT SPACE INTERPOLATION")
print("="*60)

def interpolate_latents(z1, z2, steps=8):
    """Interpolate between two latent vectors"""
    interpolations = []
    for alpha in torch.linspace(0, 1, steps):
        z = alpha * z1 + (1 - alpha) * z2
        interpolations.append(z)
    return torch.cat(interpolations)

# Create interpolation
z1 = torch.randn(1, latent_size, 1, 1, device=device)
z2 = torch.randn(1, latent_size, 1, 1, device=device)
interpolated = interpolate_latents(z1, z2, steps=8)

with torch.no_grad():
    interp_images = generator(interpolated)
    save_image(denorm(interp_images.cpu()), 'celeba_interpolation.png', nrow=8)
    
    # Display interpolation
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xticks([]); ax.set_yticks([])
    
    grid = make_grid(denorm(interp_images.cpu()), nrow=8).permute(1, 2, 0)
    grid = torch.clamp(grid, 0, 1)
    ax.imshow(grid.numpy())
    plt.title('CelebA Latent Space Interpolation')
    plt.savefig('celeba_interpolation_visualization.png', dpi=100, bbox_inches='tight')
    plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"✓ Trained on CelebA: {len(train_ds):,} images")
print(f"✓ Total epochs: {epochs}")
print(f"✓ Final models saved")
print("="*60)