import os
import argparse
import warnings

import torch
from tqdm import tqdm
from monai.transforms import LoadImage
from monai.utils import set_determinism
from torch.nn import L1Loss
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter

from brlp import const
from brlp import utils
from brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
)

# Ensure reproducibility
set_determinism(0)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Custom Dataset: Load images directly from the MRI2 folder
class MRIDataset(Dataset):
    def __init__(self, image_dir,mri_dir, normalize=True):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.endswith(('.nii', '.nii.gz'))]
        self.mri_paths = [os.path.join(image_dir, fname) for fname in os.listdir(mri_dir)
                            if fname.endswith(('.nii', '.nii.gz'))]
        self.all_paths = self.image_paths + self.mri_paths
        self.loader = LoadImage(image_only=True)
        self.normalize = normalize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.all_paths[idx]
        image = self.loader(image_path)

        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min())

        return {"image": image, "path": image_path}


if __name__ == '__main__':
    # Argument settings
    args = argparse.Namespace(
        image_dir="/home/ET/bnwu/mri->pet/PET3",  # Path to the image folder
        image_mri_dir="/home/ET/bnwu/mri->pet/MRI4",
        output_dir="/home/ET/bnwu/mri->pet/ae_output",
        aekl_ckpt=None,
        disc_ckpt=None,
        num_workers=8,
        n_epochs=1000,
        max_batch_size=2,
        batch_size=16,
        lr=1e-4,
    )

    # Data loading
    trainset = MRIDataset(image_dir=args.image_dir,mri_dir=args.image_mri_dir)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=args.num_workers,
                              batch_size=args.max_batch_size,
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)

    # Initialize models
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    # Loss functions and weights
    adv_weight = 0.025
    perceptual_weight = 0.001
    kl_weight = 1e-7

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    perc_loss_fn = PerceptualLoss(spatial_dims=3,
                                  network_type="squeeze",
                                  is_fake_3d=True,
                                  fake_3d_ratio=0.2).to(DEVICE)

    # Optimizers and gradient accumulation
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g,
                                     grad_scaler=torch.amp.GradScaler('cuda'))

    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_d,
                                     grad_scaler=torch.amp.GradScaler('cuda'))

    avgloss = utils.AverageLoss()
    writer = SummaryWriter()
    total_counter = 0

    # Training loop
    for epoch in range(args.n_epochs):
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                images = batch["image"].to(DEVICE)
                images = images.unsqueeze(1)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                # Generator loss
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

                loss_g = rec_loss + kld_loss + per_loss + gen_loss
                gradacc_g.step(loss_g, step)

            # Discriminator loss
            with torch.amp.autocast(device_type='cuda', enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                gradacc_d.step(loss_d, step)

            # Logging
            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Generator/perceptual_loss', per_loss.item())
            avgloss.put('Generator/adversarial_loss', gen_loss.item())
            avgloss.put('Generator/kl_regularization', kld_loss.item())
            avgloss.put('Discriminator/adversarial_loss', loss_d.item())

            if total_counter % 10 == 0:
                step = total_counter // 10
                avgloss.to_tensorboard(writer, step)
                utils.tb_display_reconstruction(writer, step, images[0].detach().cpu(), reconstruction[0].detach().cpu())

            total_counter += 1

        # Save the model
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(), os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
