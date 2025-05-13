import os
import argparse
import matplotlib.pyplot as plt
import torch
import random
import torch.nn.functional as F
import pandas as pd
from monai.transforms import LoadImage
import timm
from timm.utils.model_ema import ModelEmaV3
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from clip import clip

from torchvision.transforms import Normalize
from brlp import const
from brlp import utils
from brlp import networks
from brlp import (
    get_dataset_from_pd,
    sample_using_diffusion
)
import os
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset

set_determinism(0)
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, image_features, text_features):
        # Here, both image_features and text_features have shape [6, 512]

        # Cross Attention 1: Use image features as Query, text features as Key and Value
        image_to_text, _ = self.attn(image_features, text_features, text_features)

        # Cross Attention 2: Use text features as Query, image features as Key and Value
        text_to_image, _ = self.attn(text_features, image_features, image_features)

        # You can choose to combine both attention results, or return one of them
        combined_features = image_to_text + text_to_image

        return combined_features


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    _dict['context'] = torch.tensor([_dict[c] for c in const.CONDITIONING_VARIABLES]).unsqueeze(0)
    return _dict


def images_to_tensorboard(
        writer,
        epoch,
        mode,
        autoencoder,
        diffusion,
        scale_factor
):
    """
    Visualize the generation on tensorboard
    """

    for tag_i, size in enumerate(['small', 'medium', 'large']):
        context = torch.tensor([[
            (torch.randint(60, 99, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age
            (torch.randint(1, 2, (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            (torch.randint(1, 3, (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            0.567,  # (mean) cerebral cortex
            0.539,  # (mean) hippocampus
            0.578,  # (mean) amygdala
            0.558,  # (mean) cerebral white matter
            0.30 * (tag_i + 1),  # variable size lateral ventricles
        ]])

        image = sample_using_diffusion(
            autoencoder=autoencoder,
            diffusion=diffusion,
            context=context,
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_generation(
            writer=writer,
            step=epoch,
            tag=f'{mode}/{size}_ventricles',
            image=image
        )


def resample_image(input_image, target_size):
    """
    Resample an input image to a target size, while preserving the physical spacing.

    Parameters:
    - input_image: The input SimpleITK image to be resampled.
    - target_size: A tuple/list of the target size (z, y, x).

    Returns:
    - resampled_image: The resampled SimpleITK image.
    """
    # Get current image size, origin, spacing, and direction
    current_size = input_image.GetSize()
    origin = input_image.GetOrigin()
    spacing = input_image.GetSpacing()
    direction = input_image.GetDirection()

    # Compute new spacing to keep physical space consistent
    new_spacing = [s * current / target for s, current, target in zip(spacing, current_size, target_size)]

    # Create a resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)  # Set target size
    resampler.SetOutputSpacing(new_spacing)  # Set new spacing
    resampler.SetOutputOrigin(origin)  # Preserve origin
    resampler.SetOutputDirection(direction)  # Preserve direction
    resampler.SetInterpolator(sitk.sitkLinear)  # Use linear interpolation

    # Perform resampling
    resampled_image = resampler.Execute(input_image)

    return resampled_image


class MRIDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, csv_path, pet_image_dir):
        try:
            self.mri_dir = mri_dir
            self.pet_dir = pet_dir
            self.loader = LoadImage(image_only=True)
            self.pet_image_dir = pet_image_dir
            self.csv = pd.read_csv(csv_path)
            print(f"CSV loaded from {csv_path}")

            # Get all PET file names in the PET folder
            self.pet_files = [f for f in os.listdir(pet_dir) if f.endswith(".npz")]
            self.pet_image_files = [f for f in os.listdir(pet_image_dir) if f.endswith(".nii.gz")]
            print(f"Found {len(self.pet_files)} PET files in {pet_dir}")

            # Check if the corresponding MRI file exists in the MRI folder
            self.data = []
            for pet_file in self.pet_files:
                mri_file = pet_file.replace("_PET_latent.npz", "_MRI_affined.nii.gz")
                if os.path.exists(os.path.join(mri_dir, mri_file)):
                    self.data.append((pet_file, mri_file))
                else:
                    print(f"Corresponding MRI file not found: {mri_file}")

            print(f"Data size: {len(self.data)}")

        except Exception as e:
            print(f"Error in __init__: {e}")

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the data at a specified index, including MRI, PET, and contextual information.
        :param idx: Index of the data item
        :return: MRI image, PET image, and context information
        """
        # Get the PET and MRI file names
        pet_file_name, mri_file_name = self.data[idx]
        pet_image_file_name = pet_file_name.replace("_PET_latent.npz", "_PET.nii.gz")

        # Construct file paths
        pet_path = os.path.join(self.pet_dir, pet_file_name)
        mri_path = os.path.join(self.mri_dir, mri_file_name)
        pet_image_path = os.path.join(self.pet_image_dir, pet_image_file_name)

        # Load the images directly using SimpleITK
        pet_image = np.load(pet_path)
        # mri_temp = sitk.ReadImage(mri_path)
        pet_image_2 = self.loader(pet_image_path)
        pet_image_2 = (pet_image_2 - pet_image_2.min()) / (pet_image_2.max() - pet_image_2.min())
        mri_image = self.loader(mri_path)
        mri_image = (mri_image - mri_image.min()) / (mri_image.max() - mri_image.min())

        if 'latent' not in pet_image:
            raise ValueError(f"Missing 'latent' key in {pet_file_name}")

        latent = pet_image['latent']  # Ensure this is the correct key

        # Get context information
        context = self.get_context_from_csv(pet_file_name)

        # Return the data
        return {
            'mri': mri_image,
            'latent': latent,
            'context': context,
            'PET': pet_image_2
        }

    def get_context_from_csv(self, pet_file_name):
        """
        Get contextual information corresponding to the specified PET file from the CSV file.
        :param pet_file_name: PET file name
        :return: Dictionary containing contextual information
        """
        # Convert PET file name for matching in CSV
        pet_file_name = pet_file_name.replace('_PET_latent.npz', '_PET.nii.gz')

        # Find matching row in the CSV
        matched_row = self.csv[
            self.csv["image_path"].apply(lambda x: os.path.basename(x)) == pet_file_name
            ]

        # Initialize context dictionary
        context = {
            "Age": None,
            "sex": None,
            "diagnosis": None,
            "": None,
        }

        if not matched_row.empty:
            row = matched_row.iloc[0]
            row.index = row.index.str.strip()

            # Extract contextual information from the row
            context["Age"] = row["Age"]
            context["sex"] = row["Sex"]
            context["diagnosis"] = row["diagnosis"]
            context["plasma"] = row["TESTVALUE"]
        else:
            print(f"Matching row not found in CSV: {pet_file_name}")

        # Now, process context and return
        if context["Age"] is not None and context["sex"] is not None and context["diagnosis"] is not None:
            sex_str = 'male' if context["sex"] == 'M' else 'female'
            f"Age is {context['Age']},"
            context_sentence = (

                f" TESTVALUE score is {context['plasma']}"
            )  # f"Age is {context['Age']},"
            context = context_sentence

            return context

        else:
            print("Context information is incomplete. Cannot generate context.")
            return None


# Function: Process MRI images and contextual information
def process_mri_and_context(mri_image, context):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # Load the CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device)

    batch_size, _, height, width, depth = mri_image.shape

    # Process text context (using CLIP to tokenize and encode text)
    text_input = clip.tokenize(context).to(device)
    text_features = clip_model.encode_text(text_input).unsqueeze(1)  # Shape: (batch_size, 512)
    final_features = text_features

    return final_features


if __name__ == '__main__':

    args = argparse.Namespace(
        image_dir="/home/ET/bnwu/mri->pet/Latent_result1",
        mri_dir='/home/ET/bnwu/mri->pet/MRI4',
        pet_dir='/home/ET/bnwu/mri->pet/PET3',

        image_dir_val="/home/ET/bnwu/mri->pet/Latent_result2",
        mri_dir_val='/home/ET/bnwu/mri->pet/MRI5',
        pet_dir_val='/home/ET/bnwu/mri->pet/PET4',

        csv_path='/home/ET/bnwu/mri->pet/filtered_data_PET.csv',
        cache_dir="/home/ET/bnwu/mri->pet/unet_cache",  # Path to the cache directory
        output_dir="/home/ET/bnwu/mri->pet/unet_output2",  # Path to the output directory
        aekl_ckpt="/home/ET/bnwu/mri->pet/ae_output/autoencoder-ep-182.pth",
        aekl_ckpt_mri="/home/ET/bnwu/mri->pet/ae_output/autoencoder-ep-182.pth",
        # Path to the autoencoder checkpoint
        diff_ckpt=None,  # Optional: Path to the diffusion model checkpoint
        num_workers=8,  # Number of workers for DataLoader
        n_epochs=500,  # Number of training epochs
        batch_size=6,  # Batch size for training
        lr=2.5e-5,  # Learning rate
    )

    trainset = MRIDataset(
        mri_dir=args.mri_dir,  # Path to MRI images
        pet_dir=args.image_dir,  # Path to PET images
        csv_path=args.csv_path,  # Path to the CSV file
        pet_image_dir=args.pet_dir,
    )

    validset = MRIDataset(
        mri_dir=args.mri_dir_val,  # Path to MRI images
        pet_dir=args.image_dir_val,  # Path to PET images
        csv_path=args.csv_path,  # Path to the CSV file
        pet_image_dir=args.pet_dir_val,
    )

    train_loader = DataLoader(dataset=trainset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)

    valid_loader = DataLoader(dataset=validset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=False,
                              persistent_workers=True,
                              pin_memory=True)

    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE)

    mri_autoencoder = networks.init_autoencoder(args.aekl_ckpt_mri).to(DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    inerer_ema = ModelEmaV3(inferer, decay=0.9998, device='cpu', use_warmup=True)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    scaler = GradScaler()

    with torch.no_grad():
        with autocast(device_type='cuda:1', enabled=True):
            z = trainset[0]['latent']
            z2_temp = trainset[0]['PET']
            pet_image_encode, _ = autoencoder.encode(z2_temp.unsqueeze(0).unsqueeze(0).to(DEVICE))
            z2 = pet_image_encode.squeeze(0)

    z = torch.tensor(z, dtype=torch.float32)
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

    z2 = torch.tensor(z2, dtype=torch.float32)
    scale_factor2 = 1 / torch.std(z2)
    print(f"Scaling factor set to {scale_factor2}")

    writer = SummaryWriter()
    global_counter = {'train': 0, 'valid': 0}
    loaders = {'train': train_loader, 'valid': valid_loader}
    datasets = {'train': trainset, 'valid': validset}

    for epoch in range(args.n_epochs):

        for mode in loaders.keys():

            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                # print(f"Batch {step} shape: {batch['mri'].shape}")

                with autocast(device_type='cuda:1', enabled=True):

                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    mri_image, latent, context, PET = batch['mri'], batch['latent'], batch['context'], batch['PET']

                    latents = latent.to(DEVICE) * scale_factor
                    n = latents.shape[0]

                    with torch.set_grad_enabled(mode == 'train'):
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                        mri_image = mri_image.unsqueeze(1).to(DEVICE)
                        PET = PET.unsqueeze(1).to(DEVICE)
                        with torch.no_grad():
                            mri_image_encode, _ = mri_autoencoder.encode(mri_image)
                            pet_image_encode, _ = autoencoder.encode(PET)
                            pet_image_encode = pet_image_encode * scale_factor2

                        context_clip = process_mri_and_context(mri_image_encode, context)
                        # linear_layer = nn.Linear(512, 16).to(DEVICE)
                        # context_clip_8d = linear_layer(context_clip.squeeze(1).to(DEVICE))
                        # context_clip = context_clip_8d.unsqueeze(1)  # (batch,1,16)

                        # context_clip=context.unsqueeze(1).to(DEVICE) #process_mri_and_context(mri_image_encode, context)
                        latents = torch.cat([pet_image_encode, mri_image_encode], dim=1)

                        noise = torch.randn_like(latents).to(DEVICE)
                        noise[:, 3:, :, :, :] = 0
                        # latents= pet_image_encode
                        # (12,3,20,20,12)
                        # noise_pred = inferer(
                        #     inputs=latents,
                        #     diffusion_model=diffusion,
                        #     noise=noise,
                        #     timesteps=timesteps,
                        #     condition=context_clip,
                        #     mode='crossattn'
                        # )
                        
                        noise_pred = inerer_ema(
                            inputs=latents,
                            diffusion_model=diffusion,
                            noise=noise,
                            timesteps=timesteps,
                            condition=context_clip,
                            mode='crossattn'
                        )
                        loss = F.mse_loss(noise[:, :3, :, :, :].float(), noise_pred.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

        # save the model
        savepath = os.path.join(args.output_dir, f'unet-ep-{epoch}.pth')
        torch.save(diffusion.state_dict(), savepath)

        # linear_savepath = os.path.join(args.output_dir, f'linear-ep-{epoch}.pth')
        # torch.save(linear_layer.state_dict(), linear_savepath)