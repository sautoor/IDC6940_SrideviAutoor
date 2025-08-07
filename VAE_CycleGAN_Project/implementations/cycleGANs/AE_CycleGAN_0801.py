# VAE-CycleGAN © 2025 by Sridevi Autoor is licensed under CC BY-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nd/4.0/

import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image 
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
import os


from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.utils import save_image


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    """Sets the seed for reproducibility across different components."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    # torch.cuda.manual_seed(seed)  # PyTorch CUDA (single GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA (multi-GPU)

# Example usage:
seed_value = 42
set_seed(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=499, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=601, help="number of epochs of training")  ## default =100
parser.add_argument("--n_layers", type=float, default=2, help="compression levels")
parser.add_argument("--dataset_name", type=str, default="maps", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")  ## default =1, changed from 4
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")   ## default =0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=2, help="epoch from which to start lr decay")  ## default =10, 100
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")  ## default =256
parser.add_argument("--img_width", type=int, default=64, help="size of image width")    ## default =256
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default= 1, help="interval between saving model checkpoints") ## default =-1 dont' save, 1 makes save @every epoch
parser.add_argument("--n_residual_blocks", type=int, default=2, help="number of residual blocks in generator")  ## default = 9
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")

opt = parser.parse_args()
print(opt)


saved_models_file = "saved_models_AE_CycleGAN"
images_file = "images_AE_CycleGAN"
images_cycle_file = "images_AE_CycleGAN_cycle"
test_results_file = "test_results_AE_CycleGAN"


os.makedirs(f"{images_file}/epochs_{opt.n_epochs}", exist_ok=True)
os.makedirs(f"{images_cycle_file}/epochs_{opt.n_epochs}", exist_ok=True)
os.makedirs(f"{saved_models_file}/epochs_{opt.n_epochs}", exist_ok=True)
os.makedirs(f"{test_results_file}", exist_ok=True)


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load(f"{saved_models_file}/epochs_%s/G_AB_%d.pth" % (opt.n_epochs, opt.epoch)))
    G_BA.load_state_dict(torch.load(f"{saved_models_file}/epochs_%s/G_BA_%d.pth" % (opt.n_epochs, opt.epoch)))
    D_A.load_state_dict(torch.load( f"{saved_models_file}/epochs_%s/D_A_%d.pth" % (opt.n_epochs, opt.epoch)))
    D_B.load_state_dict(torch.load (f"{saved_models_file}/epochs_%s/D_B_%d.pth" % (opt.n_epochs, opt.epoch)))


else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataset_path = "../../data/%s" % opt.dataset_name
print(f"Dataset path: {dataset_path}")

# Create the dataset instance
dataset = ImageDataset(dataset_path, transforms_=transforms_, unaligned=True)

# Print dataset length and some samples info
print(f"Number of samples in dataset: {len(dataset)}")
if len(dataset) > 0:
    print(f"First sample shape: {dataset[0]['A'].shape if 'A' in dataset[0] else 'No A found'}")
    print(f"First sample shape: {dataset[0]['B'].shape if 'B' in dataset[0] else 'No B found'}")


# Training data loader
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=False, mode="test"),
    batch_size=opt.batch_size, #5,
    shuffle=False,
    num_workers=1,
)

# Add these near your other initialization code
train_history = {
    'D_loss': [],
    'G_loss': [],
    'G_GAN': [],
    'G_cycle': [],
    'G_identity': [],
    'batches': []
}

def calculate_cr(original_size, latent_size):
    """Calculate compression ratio between original and latent representation"""
    # Assuming original_size is (channels, height, width)
    original_elements = original_size[1] * original_size[2]
   
    # Calculate compression ratio
    cr = original_elements / latent_size
    return cr

# You can call this with your image dimensions and latent dimension
latent_shape =input_shape[1]//(2**(opt.n_layers))
latent_shape = latent_shape**2  # Adjusted to match the latent dimension used in the model

cr = calculate_cr(input_shape, latent_shape)
print(f"Compression Ratio: {cr:.2f}:1")



def plot_learning_curves(history, save_path="learning_curves.png"):
    plt.figure(figsize=(12, 8))
    
    # Plot Generator losses
    plt.subplot(2, 1, 1)
    plt.plot(history['batches'], history['G_loss'], label='Total G Loss')
    plt.plot(history['batches'], history['G_GAN'], label='GAN Loss')
    plt.plot(history['batches'], history['G_cycle'], label='Cycle Loss')
    plt.plot(history['batches'], history['G_identity'], label='Identity Loss')
    plt.title('Generator Learning Curves')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Discriminator loss
    plt.subplot(2, 1, 2)
    plt.plot(history['batches'], history['D_loss'], label='D Loss', color='red')
    plt.title('Discriminator Learning Curve')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


#########################


##############################################
def save_generated_images(epoch_num):
    """Saves generated samples in a 4x4 grid with clear labels"""
    G_AB.eval()
    G_BA.eval()

    try:
        # Get a batch of test images
        batch = next(iter(val_dataloader))
        # real_A = batch["A"].type(Tensor)      # Real A (Domain A)
        # real_B = batch["B"].type(Tensor)     # Real B (Domain B)
        
        real_A = batch["A"].cuda()      # Real A (Domain A)
        real_B = batch["B"].cuda()     # Real B (Domain B)

        
        with torch.no_grad():
            fake_B = G_AB(real_A)           # Fake B (A → B)
            fake_A = G_BA(real_B)           # Fake A (B → A)
            
            cycle_A = G_BA(fake_B)           # Fake B → Cycle-Consistent A          
            cycle_B= G_AB(cycle_A)           # cycle_A -> Cycle-Consistent B

        # Denormalize images
        def denorm(tensor):
            return torch.clamp(tensor * 0.5 + 0.5, 0, 1)


        _real_A = denorm(real_A[:4])
        _fake_B = denorm(fake_B[:4])
        _real_B = denorm(real_B[:4])

        _fake_A = denorm(fake_A[:4])        
        _cycle_A = denorm(cycle_A[:4])
        _cycle_B = denorm(cycle_B[:4])
        
        #################
               # Arrange in 4x4 grid:
        # Row 1: Real A (4 images)
        # Row 2: Fake B (A→B translations)
        # Row 3: Real B (4 images)
        # Row 4: Fake A (B→A translations)
        grid_rows1 = [
            _real_A,     # Row 1: Real A
            _fake_B,    # Row 2: Fake B (generated from real A)
            _real_B,    # Row 3: Real B
            _fake_A      # Row 4: Fake A (generated from real B)
        ]
        grid1 = torch.cat(grid_rows1, dim=0)
        
        # Save the grid

        # In your sample_images funAion:
        # save_colored_image( grid1, f"images_A2B_general/{opt.dataset_name}/{batches_done}.png", nrow=4 )

        save_image(
            grid1,
            f"{images_file}/epochs_{opt.n_epochs}/{epoch_num}.png",
            nrow=4,       # 4 columns (4 images per row)
            normalize=False,
            padding=2,
            pad_value=1.0  # White padding between images
        )
        ##############

        # Arrange in 4x4 grid:
        # Row 1: Real A (4 images)
        # Row 2: Fake B (A→B translations)
         # Row 3: Cycle-Consistent A (fake B → A)
        # Row 4: Fake A (B→A translations)
        grid_rows2 = [
            _real_A,      # Row 1: Real A
            _fake_B,     # Row 2: Fake B (A → B)
            _cycle_A,     # Row 3: Cycle-A (B → A cycle)
            _cycle_B     # Row 4: cycle-B (cycle A → cycle B)
        ]
        grid2= torch.cat(grid_rows2, dim=0)
        # grid2 = add_text_column(grid2, ["Real A", "Fake B", "Cycle A", "Cycle B"])

        #Save the grid
        save_image(
            grid2,
            f"{images_cycle_file}/epochs_{opt.n_epochs}/{epoch_num}.png",
            nrow=4,       # 4 columns (4 images per row)
            normalize=False,
            padding=2,
            pad_value=1.0  # White padding between images
        )
        

    except Exception as e:
        print(f"Error saving sample images: {str(e)}")

    return real_A, real_B, fake_A, fake_B, cycle_A, cycle_B

###########################


def vae_loss(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x, reduAion='sum')  # ReconstruAion loss
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())  # KL divergence
    return BCE + KLD

########################################
# ----------
#  Training
# ----------

prev_time = time.time()


for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = torch.ones(real_A.size(0), *D_A.output_shape, 
                  dtype=torch.float32, device=real_A.device)
        # fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = torch.tensor(np.zeros((real_A.size(0), *D_A.output_shape)), dtype=torch.float32, device='cuda', requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle  #+ opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

    
        # Store losses
        train_history['D_loss'].append(loss_D.item())
        train_history['G_loss'].append(loss_G.item())
        train_history['G_GAN'].append(loss_GAN.item())
        train_history['G_cycle'].append(loss_cycle.item())
        train_history['G_identity'].append(loss_identity.item())
        train_history['batches'].append(batches_done)


        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )


    if ((epoch + 1) % 10 == 0 or (epoch + 1) == opt.epoch or (epoch + 1) == opt.n_epochs  or (epoch + 1) > (opt.n_epochs-5)):
     # After training completes
        plot_learning_curves(train_history, f"{images_file}/epochs_{opt.n_epochs}/final_learning_curves.png")
        val_real_A, val_real_B, val_fake_A, val_fake_B, val_cycle_A, val_cycle_B = save_generated_images(epoch)  # 

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # After training completes
    plot_learning_curves(train_history, "final_learning_curves.png")

    if opt.checkpoint_interval != -1 and (((epoch +1) %100 ==0) or (epoch +1) == opt.n_epochs): #(#(epoch) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), f"{saved_models_file}/epochs_%s/G_AB_%d.pth" % (opt.n_epochs, epoch))
        torch.save(G_BA.state_dict(), f"{saved_models_file}/epochs_%s/G_BA_%d.pth" % (opt.n_epochs, epoch))
        torch.save(D_A.state_dict(), f"{saved_models_file}/epochs_%s/D_A_%d.pth" % (opt.n_epochs, epoch))
        torch.save(D_B.state_dict(), f"{saved_models_file}/epochs_%s/D_B_%d.pth" % (opt.n_epochs, epoch))

    # In training loop (add after optimizer steps):
    torch.nn.utils.clip_grad_norm_(G_AB.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(G_BA.parameters(), 1.0)

###########################################################################

# Initialize metrics
fid = FrechetInceptionDistance(feature=2048).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
# Add these near your other metric initializations
mse = MeanSquaredError().to(device)
psnr = PeakSignalNoiseRatio().to(device)

# Function to calculate metrics between real and fake images
# Modify your calculate_metrics function:
@torch.no_grad()
def calculate_metrics(real, fake):
    """Calculate various metrics between real and fake images"""
    metrics = {}
    

    """Handle batch size mismatches by trimming to smaller size"""
    min_batch = min(real.shape[0], fake.shape[0])
    real = real[:min_batch]
    fake = fake[:min_batch]

    # Denormalize images (assuming they're in [-1, 1] range)
    real_denorm = (real + 1) / 2  # Scale to [0, 1]
    fake_denorm = (fake + 1) / 2
    
    # Reset metrics
    mse.reset()
    psnr.reset()
    ssim.reset()
    fid.reset()
    
    # Calculate metrics
    metrics['mse'] = mse(real_denorm, fake_denorm).item()
    metrics['psnr'] = psnr(real_denorm, fake_denorm).item()
    metrics['ssim'] = ssim(real_denorm, fake_denorm).item()
    
    # For FID, we need uint8 images in 0-255 range
    real_uint8 = (real_denorm * 255).byte()
    fake_uint8 = (fake_denorm * 255).byte()
    
    # Update FID
    fid.update(real_uint8, real=True)
    fid.update(fake_uint8, real=False)
    metrics['fid'] = fid.compute().item()

    # Convert all metrics to Python floats
    # return {k: float(v) if hasattr(v, 'item') else float(v) for k, v in metrics.items()}
    
    return metrics

##############################
# dictionary to track epoch metrics
epoch_metrics = {
    'train': {'mse': [], 'psnr': [], 'ssim': [], 'fid': []},
    'val': {'mse': [], 'psnr': [], 'ssim': [], 'fid': []}
}


    # Initialize epoch metrics
train_metrics_A = {'mse': 0, 'psnr': 0, 'ssim': 0, 'fid':0, 'count': 0}
val_metrics_A =  {'mse': 0, 'psnr': 0, 'ssim': 0, 'fid': 0, 'count': 0}  

train_metrics_B = {'mse': 0, 'psnr': 0, 'ssim': 0, 'fid':0, 'count': 0}
val_metrics_B =  {'mse': 0, 'psnr': 0, 'ssim': 0, 'fid': 0, 'count': 0} 

train_metrics_B = {'mse': 0, 'psnr': 0, 'ssim': 0, 'fid':0, 'count': 0}
val_metrics_B =  {'mse': 0, 'psnr': 0, 'ssim': 0, 'fid': 0, 'count': 0} 

train_metrics_A = calculate_metrics(real_A, fake_A)
val_metrics_A = calculate_metrics(val_real_A, val_fake_A)

train_metrics_B = calculate_metrics(real_B, fake_B)
val_metrics_B = calculate_metrics(val_real_B, val_fake_B)

cycle_metrics_A = calculate_metrics(real_A, val_cycle_A)
cycle_metrics_B = calculate_metrics(real_B, val_cycle_B)


print(train_metrics_A)
print(val_metrics_A)

print(train_metrics_B)
print(val_metrics_B)

print(cycle_metrics_A)
print(cycle_metrics_B)

# Save metrics to file
# metrics_path = f"{model_config['test_results_file']}/epochs_{model_config['n_epochs']}_metrics.txt"
metrics_path = f"{test_results_file}/epochs_{opt.n_epochs}_metrics.txt"
with open(metrics_path, "w") as f:
    # f.write("Epoch\tTrain MSE\tTrain PSNR\tTrain SSIM\tTrain FID\tVal MSE\tVal PSNR\tVal SSIM\tVal FID\n")
    f.write(f"Compression Ratio: {cr:.2f}:1\n\n")

    ################ TRAIN A and VAL A

    f.write("Train_A and Val_A\n")
    f.write("Epoch\tTrain MSE\tTrain PSNR\tTrain SSIM\tTrain FID\tVal MSE \tVal PSNR\tVal SSIM\tVal FID\n")

    f.write(f"{epoch}\t")
    f.write(f"{train_metrics_A['mse']:.4f}\t\t")
    f.write(f"{train_metrics_A['psnr']:.2f}\t\t")
    f.write(f"{train_metrics_A['ssim']:.4f}\t\t")
    f.write(f"{train_metrics_A['fid']:.2f}\t\t")
    

    f.write(f"{val_metrics_A['mse']:.4f}\t\t")
    f.write(f"{val_metrics_A['psnr']:.2f}\t\t")
    f.write(f"{val_metrics_A['ssim']:.4f}\t\t")
    f.write(f"{val_metrics_A['fid']:.2f}\n\n")
   
    ####### TRAIN B and VAL B 
    f.write("Train_B and Val_B\n")
    f.write("Epoch\tTrain MSE\tTrain PSNR\tTrain SSIM\tTrain FID\tVal MSE \tVal PSNR\tVal SSIM\tVal FID\n")
    f.write(f"{epoch}\t")
    f.write(f"{train_metrics_B['mse']:.4f}\t\t")
    f.write(f"{train_metrics_B['psnr']:.2f}\t\t")
    f.write(f"{train_metrics_B['ssim']:.4f}\t\t")
    f.write(f"{train_metrics_B['fid']:.2f}\t\t")
    

    f.write(f"{val_metrics_B['mse']:.4f}\t\t")
    f.write(f"{val_metrics_B['psnr']:.2f}\t\t")
    f.write(f"{val_metrics_B['ssim']:.4f}\t\t")
    f.write(f"{val_metrics_B['fid']:.2f}\n\n")

    ####### TRAIN A and CYCLE A
    f.write("Train_A and Cycle_A\n")
    f.write("Epoch\tTrain MSE\tTrain PSNR\tTrain SSIM\tTrain FID\tCyc MSE \tCyc PSNR\tCyc SSIM\tCyc FID\n")

    f.write(f"{epoch}\t")
    f.write(f"{train_metrics_A['mse']:.4f}\t\t")
    f.write(f"{train_metrics_A['psnr']:.2f}\t\t")
    f.write(f"{train_metrics_A['ssim']:.4f}\t\t")
    f.write(f"{train_metrics_A['fid']:.2f}\t\t")
    

    f.write(f"{cycle_metrics_A['mse']:.4f}\t\t")
    f.write(f"{cycle_metrics_A['psnr']:.2f}\t\t")
    f.write(f"{cycle_metrics_A['ssim']:.4f}\t\t")
    f.write(f"{cycle_metrics_A['fid']:.2f}\n\n")

    ############ TRAIN B and CYCLE B

    f.write("Train_B and Cycle_B\n")
    f.write("Epoch\tTrain MSE\tTrain PSNR\tTrain SSIM\tTrain FID\tCyc MSE \tCyc PSNR\tCyc SSIM\tCyc FID\n")
    f.write(f"{epoch}\t")
    f.write(f"{train_metrics_B['mse']:.4f}\t\t")
    f.write(f"{train_metrics_B['psnr']:.2f}\t\t")
    f.write(f"{train_metrics_B['ssim']:.4f}\t\t")
    f.write(f"{train_metrics_B['fid']:.2f}\t\t")
    

    f.write(f"{cycle_metrics_B['mse']:.4f}\t\t")
    f.write(f"{cycle_metrics_B['psnr']:.2f}\t\t")
    f.write(f"{cycle_metrics_B['ssim']:.4f}\t\t")
    f.write(f"{cycle_metrics_B['fid']:.2f}\n\n")