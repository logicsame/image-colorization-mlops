import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
import pytorch_lightning as pl
from torchsummary import summary
from src.imagecolorization.conponents.model_building import Generator, Critic
from src.imagecolorization.conponents.data_tranformation import ImageColorizationDataset
from src.imagecolorization.logging import logger
from src.imagecolorization.entity.config_entity import ModelTrainerConfig
import os 



def lab_to_rgb(L, ab):
    L = L * 100
    ab = (ab - 0.5) * 128 * 2
    Lab = torch.cat([L, ab], dim = 2).numpy()
    rgb_img = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_img.append(img_rgb)
        
    return np.stack(rgb_img, axis = 0)



import matplotlib.pyplot as plt
def display_progress(cond, real, fake, current_epoch = 0, figsize=(20,15)):
    """
    Save cond, real (original) and generated (fake)
    images in one panel 
    """
    cond = cond.detach().cpu().permute(1, 2, 0)   
    real = real.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    
    images = [cond, real, fake]
    titles = ['input','real','generated']
    print(f'Epoch: {current_epoch}')
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for idx,img in enumerate(images):
        if idx == 0:
            ab = torch.zeros((224,224,2))
            img = torch.cat([images[0]* 100, ab], dim=2).numpy()
            imgan = lab2rgb(img)
        else:
            imgan = lab_to_rgb(images[0],img)
        ax[idx].imshow(imgan)
        ax[idx].axis("off")
    for idx, title in enumerate(titles):    
        ax[idx].set_title('{}'.format(title))
    plt.show()


class CWGAN(pl.LightningModule):
    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10):
        super().__init__()
        self.save_hyperparameters()
        self.display_step = display_step
        self.generator = Generator(in_channels, out_channels)
        self.critic = Critic(in_channels + out_channels)
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.recon_criterion = nn.L1Loss()
        self.generator_losses, self.critic_losses = [], []
        self.automatic_optimization = False  # Disable automatic optimization
        
    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.9))
        optimizer_C = optim.Adam(self.critic.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.9))
        return [optimizer_C, optimizer_G]
    
    def generator_step(self, real_images, conditioned_images, optimizer_G):
        # WGAN has only a reconstruction loss
        optimizer_G.zero_grad()
        fake_images = self.generator(conditioned_images)
        recon_loss = self.recon_criterion(fake_images, real_images)
        recon_loss.backward()
        optimizer_G.step()
        self.generator_losses.append(recon_loss.item())
        
    def critic_step(self, real_images, conditioned_images, optimizer_C):
        optimizer_C.zero_grad()
        fake_images = self.generator(conditioned_images)
        fake_logits = self.critic(fake_images, conditioned_images)
        real_logits = self.critic(real_images, conditioned_images)
        
        # Compute the loss for the critic
        loss_C = real_logits.mean() - fake_logits.mean()

        # Compute the gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True).to(real_images.device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
        interpolated_logits = self.critic(interpolated, conditioned_images)
        
        gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated,
                                        grad_outputs=torch.ones_like(interpolated_logits), create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(len(gradients), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradients_penalty
        
        # Compute the R1 regularization loss
        r1_reg = gradients.pow(2).sum(1).mean()
        loss_C += self.lambda_r1 * r1_reg

        # Backpropagation
        loss_C.backward()
        optimizer_C.step()
        self.critic_losses.append(loss_C.item())
    
    def training_step(self, batch, batch_idx):
        real, condition = batch
        optimizer_C, optimizer_G = self.optimizers()  # Access optimizers
    
        # Update the critic
        self.critic_step(real, condition, optimizer_C)
    
        # Update the generator
        self.generator_step(real, condition, optimizer_G)
    
        # Logging and saving models
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        if self.current_epoch % self.display_step == 0 and batch_idx == 0:
            fake = self.generator(condition).detach()
            logger.info(f"Epoch {self.current_epoch}: Generator loss: {gen_mean}, Critic loss: {crit_mean}")
            display_progress(condition[0], real[0], fake[0], self.current_epoch)
    

class SaveEveryNepochs(pl.callbacks.Callback):
    def __init__(self, save_interval, trainer):
        super().__init__()
        self.save_interval = save_interval
        self.trainer = trainer
    
    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_interval == 0:
            epoch = trainer.current_epoch + 1
            generator_path = os.path.join(self.trainer.config.root_dir, f"cwgan_generator_epoch_{epoch}.pt")
            critic_path = os.path.join(self.trainer.config.root_dir, f"cwgan_critic_epoch_{epoch}.pt")
            
            torch.save(pl_module.generator.state_dict(), generator_path)
            torch.save(pl_module.critic.state_dict(), critic_path)
            logger.info(f"Models saved at epoch {epoch}")

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def load_datasets(self):
        self.train_dataset = torch.load(self.config.train_data_path)
        self.test_dataset = torch.load(self.config.test_data_path)
    
    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
        )
    
    def initialize_model(self):
        self.model = CWGAN(
            in_channels=1,
            out_channels=2,
            learning_rate=self.config.LEARNING_RATE,
            lambda_recon=self.config.LAMBDA_RECON,
            display_step=self.config.DISPLAY_STEP,
        )
        
    def train_model(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.config.root_dir,
            filename='cwgan-{epoch:02d}-{generator_loss:.2f}',
            save_top_k=-1,  
            verbose=True
        )
        
        save_every_50_epochs = SaveEveryNepochs(50, self)
        
        trainer = pl.Trainer(
            max_epochs=self.config.EPOCH,
            callbacks=[checkpoint_callback, save_every_50_epochs],
        )
        
        trainer.fit(self.model, self.train_dataloader)
        
    def save_model(self):
        trained_model_dir = self.config.root_dir
        os.makedirs(trained_model_dir, exist_ok=True)
    
        generator_path = os.path.join(trained_model_dir, "cwgan_generator_final.pt")
        critic_path = os.path.join(trained_model_dir, "cwgan_critic_final.pt")
    
        torch.save(self.model.generator.state_dict(), generator_path)
        torch.save(self.model.critic.state_dict(), critic_path)
        logger.info(f"Final models saved at {trained_model_dir}")