import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import clip
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FusionAttention(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8):
        super(FusionAttention, self).__init__()
        # Define multi-head attention layer
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(feature_dim, feature_dim)  # Optional fully connected layer to adjust dimensions

    def forward(self, mri_features, text_features):
        # Combine MRI and text features (concatenate along the feature dimension)
        combined_features = torch.cat([mri_features, text_features], dim=1)  # Shape: (batch_size, 1024)

        # Apply multi-head attention
        attn_output, _ = self.attn(combined_features, combined_features,
                                   combined_features)  # Shape: (batch_size, 1024, 512)

        # Optionally, pass through a fully connected layer for further refinement
        output = self.fc(attn_output)

        return output


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
    def __init__(self, mri_dir, pet_dir, csv_path):
        try:
            self.mri_dir = mri_dir
            self.pet_dir = pet_dir
            self.csv = pd.read_csv(csv_path)
            print(f"CSV loaded from {csv_path}")

            # Get all PET file names in the PET folder
            self.pet_files = [f for f in os.listdir(pet_dir) if f.endswith(".npz")]
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

        # Construct file paths
        pet_path = os.path.join(self.pet_dir, pet_file_name)
        mri_path = os.path.join(self.mri_dir, mri_file_name)

        # Load the images directly using SimpleITK
        pet_image = np.load(pet_path)
        mri_temp = sitk.ReadImage(mri_path)
        # target_size = [170, 256, 256]  # Define target size
        # mri_temp2 = resample_image(mri_temp, target_size)
        mri_image = sitk.GetArrayFromImage(mri_temp)

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
            "mmse": None,
        }

        if not matched_row.empty:
            row = matched_row.iloc[0]
            row.index = row.index.str.strip()

            # Extract contextual information from the row
            context["Age"] = row["Age"]
            context["sex"] = row["Sex"]
            context["diagnosis"] = row["diagnosis"]
            context["mmse"] = row["MMSE"]
        else:
            print(f"Matching row not found in CSV: {pet_file_name}")

        # Now, process context and return
        if context["Age"] is not None and context["sex"] is not None and context["diagnosis"] is not None:
            sex_str = 'male' if context["sex"] == 'M' else 'female'
            context_sentence = (
                f"The patient is {context['Age']} years old, which provides insight into their potential cognitive development. "
                f"The patient is {sex_str}, indicating their gender. The diagnosis is {context['diagnosis']}, "
                f"which classifies the patient's cognitive condition, and their MMSE score is {context['mmse']}, "
                f"reflecting the current state of their cognitive function."
            )
            context = context_sentence

            return context
        else:
            print("Context information is incomplete. Cannot generate context.")
            return None


# Function: Process MRI images and contextual information
def process_mri_and_context(mri_image, context):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device)

    batch_size, _, height, width, depth = mri_image.shape

    # Process text context (using CLIP to tokenize and encode text)
    text_input = clip.tokenize(context).to(device)
    text_features = clip_model.encode_text(text_input)  # Shape: (batch_size, 512)

    # Initialize list to store MRI slice features
    mri_features = []

    # Process each slice of the MRI images
    for i in range(depth):
        # Extract the i-th slice
        slice_image = mri_image[:, :, :, :, i]  # Shape: (batch_size, 1, height, width)

        # Normalize slice to the range [0, 1] as required
        slice_image = (slice_image - slice_image.min()) / (slice_image.max() - slice_image.min())
        # Convert slice to the range [-1, 1] for CLIP
        slice_image = 2 * slice_image - 1  # Normalize to [-1, 1]

        # Resize slice to (224, 224)
        slice_image = F.interpolate(slice_image, size=(224, 224), mode="bilinear", align_corners=False)

        # CLIP expects 3-channel input; repeat the single-channel slice 3 times
        slice_image = slice_image.repeat(1, 3, 1, 1)  # Shape: (batch_size, 3, 224, 224)

        # Move slice to device and encode it with CLIP
        slice_image = slice_image.to(device)
        slice_features = clip_model.encode_image(slice_image)  # Shape: (batch_size, 512)

        mri_features.append(slice_features)

    # Stack all slice features
    mri_features = torch.stack(mri_features, dim=1)  # Shape: (batch_size, depth, 512)

    # Average the slice features to obtain a single feature for the entire MRI
    avg_mri_features = mri_features.mean(dim=1)  # Shape: (batch_size, 512)

    fc_layer = nn.Linear(512, 256).to(device)
    text_features = fc_layer(text_features.to(device))
    mri_features = fc_layer(avg_mri_features.to(device))

    # Apply Fusion Attention
    fusion_attention = FusionAttention(feature_dim=512, num_heads=8).to(device)
    final_features = fusion_attention(mri_features, text_features)  # Shape: (batch_size, 512)

    final_features = final_features.unsqueeze(1)
    final_features = fc_layer(final_features)  # Shape: [15,1, 256]

    return final_features


if __name__ == '__main__':

    args = argparse.Namespace(
        image_dir="/blue/kgong/gongyuxin/taupet/ADNI_data/Latent_result",
        mri_dir='/blue/kgong/gongyuxin/taupet/ADNI_data/MRI5',
        pet_dir='/blue/kgong/gongyuxin/taupet/ADNI_data/PET3',

        image_dir_val="/blue/kgong/gongyuxin/taupet/ADNI_data/Latent_result2",
        mri_dir_val='/blue/kgong/gongyuxin/taupet/ADNI_data/MRI6',
        pet_dir_val='/blue/kgong/gongyuxin/taupet/ADNI_data/PET4',

        csv_path='/orange/kgong/Data/ADNI/filtered_data_PET.csv',
        cache_dir="/blue/kgong/gongyuxin/taupet/BrLP/scripts/unet_cache",  # Path to the cache directory
        output_dir="/blue/kgong/gongyuxin/taupet/BrLP/scripts/unet_output",  # Path to the output directory
        aekl_ckpt="/blue/kgong/gongyuxin/taupet/BrLP/scripts/ae_output/autoencoder-ep-150.pth",
        # Path to the autoencoder checkpoint
        diff_ckpt=None,  # Optional: Path to the diffusion model checkpoint
        num_workers=8,  # Number of workers for DataLoader
        n_epochs=500,  # Number of training epochs
        batch_size=15,  # Batch size for training
        lr=2.5e-5,  # Learning rate
    )

    trainset = MRIDataset(
        mri_dir=args.mri_dir,  # Path to MRI images
        pet_dir=args.image_dir,  # Path to PET images
        csv_path=args.csv_path  # Path to the CSV file
    )

    validset = MRIDataset(
        mri_dir=args.mri_dir_val,  # Path to MRI images
        pet_dir=args.image_dir_val,  # Path to MRI images
        csv_path=args.csv_path  # Path to the CSV file
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

    mri_autoencoder = networks.init_autoencoder(None).to(DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    scaler = GradScaler()

    with torch.no_grad():
        with autocast(device_type='cuda', enabled=True):
            z = trainset[0]['latent']

    z = torch.tensor(z, dtype=torch.float32)
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

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

                with autocast(device_type='cuda', enabled=True):

                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    mri_image, latent, context = batch['mri'], batch['latent'], batch['context']
                    # print(
                    #     f"Batch {step} - MRI shape: {mri_image.shape}, Latent shape: {latent.shape}, Context: {len(context)}")

                    latents = latent.to(DEVICE) * scale_factor
                    context = context
                    n = latents.shape[0]

                    with torch.set_grad_enabled(mode == 'train'):
                        noise = torch.randn_like(latents).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                        mri_image = mri_image.unsqueeze(1).half().to(DEVICE)
                        context_clip = process_mri_and_context(mri_image, context)

                        # print('context#############', context_clip.shape)
                        # print('latents.permute(0, 4, 1, 2, 3)#############', latents.permute(0, 4, 1, 2, 3).shape)
                        # print('noise.permute(0, 4, 1, 2, 3)#############', noise.permute(0, 4, 1, 2, 3).shape)

                        noise_pred = inferer(
                            inputs=latents,
                            diffusion_model=diffusion,
                            noise=noise,
                            timesteps=timesteps,
                            condition=context_clip,
                            mode='crossattn'
                        )

                        loss = F.mse_loss(noise.float(), noise_pred.float())

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

            # # visualize results
            # images_to_tensorboard(
            #     writer=writer,
            #     epoch=epoch,
            #     mode=mode,
            #     autoencoder=autoencoder,
            #     diffusion=diffusion,
            #     scale_factor=scale_factor
            # )

        # save the model
        savepath = os.path.join(args.output_dir, f'unet-ep-{epoch}.pth')
        torch.save(diffusion.state_dict(), savepath)