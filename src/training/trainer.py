import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os

class GANTrainer:
    def __init__(self, generator, discriminator, dataloader, config, device):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.config = config
        self.device = device
        
        self.latent_size = config['model']['latent_size']
        self.batch_size = config['training']['batch_size']
        
        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=config['training']['lr_g'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['training']['lr_d'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        # Fixed latent for consistent samples
        self.fixed_latent = torch.randn(64, self.latent_size, 1, 1, device=self.device)
        
    def train_discriminator(self, real_images):
        self.opt_d.zero_grad()
        
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Train with real images
        real_preds = self.discriminator(real_images)
        real_loss = F.binary_cross_entropy(real_preds, real_labels)
        real_score = torch.mean(real_preds).item()
        
        # Generate fake images
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        
        # Train with fake images
        fake_preds = self.discriminator(fake_images.detach())
        fake_loss = F.binary_cross_entropy(fake_preds, fake_labels)
        fake_score = torch.mean(fake_preds).item()
        
        loss_d = real_loss + fake_loss
        loss_d.backward()
        self.opt_d.step()
        
        return loss_d.item(), real_score, fake_score
    
    def train_generator(self):
        self.opt_g.zero_grad()
        
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        fake_preds = self.discriminator(fake_images)
        fake_labels = torch.ones(self.batch_size, 1, device=self.device)
        
        loss_g = F.binary_cross_entropy(fake_preds, fake_labels)
        loss_g.backward()
        self.opt_g.step()
        
        return loss_g.item()
    
    def train_epoch(self, epoch):
        epoch_loss_g = 0
        epoch_loss_d = 0
        epoch_real_score = 0
        epoch_fake_score = 0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        for real_images, _ in pbar:
            real_images = real_images.to(self.device)
            
            # Train discriminator
            loss_d, real_score, fake_score = self.train_discriminator(real_images)
            
            # Train generator
            loss_g = self.train_generator()
            
            # Accumulate
            epoch_loss_g += loss_g
            epoch_loss_d += loss_d
            epoch_real_score += real_score
            epoch_fake_score += fake_score
            num_batches += 1
            
            pbar.set_postfix({
                'loss_g': f'{loss_g:.3f}',
                'loss_d': f'{loss_d:.3f}',
                'D(real)': f'{real_score:.3f}',
                'D(fake)': f'{fake_score:.3f}'
            })
        
        # Return average metrics
        return {
            'loss_g': epoch_loss_g / num_batches,
            'loss_d': epoch_loss_d / num_batches,
            'real_score': epoch_real_score / num_batches,
            'fake_score': epoch_fake_score / num_batches
        }
    
    def train(self):
        history = {
            'losses_g': [],
            'losses_d': [],
            'real_scores': [],
            'fake_scores': []
        }
        
        for epoch in range(self.config['training']['epochs']):
            metrics = self.train_epoch(epoch)
            
            # Record history
            history['losses_g'].append(metrics['loss_g'])
            history['losses_d'].append(metrics['loss_d'])
            history['real_scores'].append(metrics['real_score'])
            history['fake_scores'].append(metrics['fake_score'])
            
            # Log
            print(f"Epoch [{epoch+1}/{self.config['training']['epochs']}]: "
                  f"loss_g: {metrics['loss_g']:.4f}, loss_d: {metrics['loss_d']:.4f}, "
                  f"real_score: {metrics['real_score']:.4f}, fake_score: {metrics['fake_score']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
        
        return history
    
    def save_checkpoint(self, epoch):
        checkpoint_dir = self.config['logging']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(self.generator.state_dict(), 
                  os.path.join(checkpoint_dir, f'G_epoch_{epoch}.pth'))
        torch.save(self.discriminator.state_dict(), 
                  os.path.join(checkpoint_dir, f'D_epoch_{epoch}.pth'))
        
        print(f"✓ Checkpoint saved at epoch {epoch}")