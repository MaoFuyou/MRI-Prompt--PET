import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from brlp import networks, utils
from brlp import (
    get_dataset_from_pd,
    sample_using_diffusion
)
import numpy as np
import torch.nn as nn
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
import clip

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

def save_image_with_metadata(generated_image, path):
    # Step 1: Read the original image and extract its metadata (spacing, origin, direction)
    print(sitk.ReadImage(path).GetSize())
    original_image = sitk.ReadImage(path)[:, :, :, 0]
    spacing = original_image.GetSpacing()[:3]
    origin = original_image.GetOrigin()[:3]
    direction = original_image.GetDirection()[:9]

    # Step 2: Convert the generated image (NumPy array) to a SimpleITK image
    image_to_save = sitk.GetImageFromArray(generated_image.cpu().numpy().astype(np.float32).transpose(2, 1, 0))

    # Step 3: Set the metadata (spacing, origin, direction) for the generated image
    image_to_save.SetSpacing(spacing)
    image_to_save.SetOrigin(origin)
    image_to_save.SetDirection(direction)

    # Step 4: Return the SimpleITK image with metadata (no saving to disk)
    return image_to_save


# Define the test function

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
    def __init__(self, mri_dir, pet_dir, csv_path,pet_dir2):
        try:
            self.mri_dir = mri_dir
            self.pet_dir = pet_dir
            self.pet_dir2= pet_dir2

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
        mri_image = sitk.GetArrayFromImage(mri_temp)
        mri_image = (mri_image - mri_image.min()) / (mri_image.max() - mri_image.min())

        if 'latent' not in pet_image:
            raise ValueError(f"Missing 'latent' key in {pet_file_name}")

        latent = pet_image['latent']  # Ensure this is the correct key
        Age='81.2'
        Sex= 'M'
        diagnosis= 'CN'
        MMSE='30'

        # Get context information
        context = self.get_context_from_csv(pet_file_name, Age, Sex, diagnosis, MMSE)

        pet_path = os.path.join(self.pet_dir2, pet_file_name.replace("_latent.npz", ".nii.gz"))


        # Return the data
        return {
            'mri': mri_image,
            'latent': latent,
            'context': context,
            'pet_path': pet_path,
        }

    def get_context_from_csv(self, pet_file_name,Age, Sex,diagnosis,MMSE):
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
            context["Age"] = Age
            context["sex"] = Sex
            context["diagnosis"] = diagnosis
            context["mmse"] = MMSE
        else:
            print(f"Matching row not found in CSV: {pet_file_name}")

        # Now, process context and return
        if context["Age"] is not None and context["sex"] is not None and context["diagnosis"] is not None:
            sex_str = 'male' if context["sex"] == 'M' else 'female'
            context_sentence = (
                f"The patient is {context['Age']},"
                f"The patient is {sex_str}. The diagnosis is {context['diagnosis']}, "
                f" MMSE score is {context['mmse']}"
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

    #######
    mri_image = mri_image.to(device)
    conv1 = nn.Conv3d(in_channels=3, out_channels=512, kernel_size=3, padding=1).to(device)
    mri_image = conv1(mri_image)
    avg_pool = nn.AdaptiveAvgPool3d(output_size=1)  # Output shape will be [batch_size, channels, 1, 1, 1]
    pooled_mri = avg_pool(mri_image)  # Shape: [5, 512, 1, 1, 1]

    # Step 2: Flatten the pooled MRI features
    flattened_mri = pooled_mri.view(pooled_mri.size(0), -1)  # Shape: [5, 512]

    # Step 3: Concatenate text and MRI features along the feature dimension
    # combined_features = torch.cat((flattened_mri, text_features), dim=1)  # Shape: [5, 1024]
    cross_attention = CrossAttention(dim=512, num_heads=8).to(device)

    final_features = cross_attention(flattened_mri, text_features).unsqueeze(1)  # [5,1, 512]

    return final_features


def test(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    testset = MRIDataset(
        mri_dir=args.mri_dir,  # Path to MRI images
        pet_dir=args.image_dir,  # Path to PET images
        csv_path=args.csv_path,
        pet_dir2= args.pet_dir

    )

    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True
    )

    # Load models
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE)
    mri_autoencoder = networks.init_autoencoder(args.aekl_ckpt_mri).to(DEVICE)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    with torch.no_grad():
        with autocast(device_type='cuda', enabled=True):
            z = testset[0]['latent']

    z = torch.tensor(z, dtype=torch.float32)
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

    # Set model to evaluation mode
    diffusion.eval()
    epoch=0

    with torch.no_grad():

        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
        progress_bar.set_description(f"Testing epoch {epoch}")

        for step, batch in progress_bar:
            # Extract data from the batch
            mri_image, latent, context, path = batch['mri'], batch['latent'], batch['context'],batch['pet_path']

            # Print the batch shape for debugging
            # print(
            #     f"Testing - Batch {step} - MRI shape: {mri_image.shape}, Latent shape: {latent.shape}, Context: {len(context)}")

            # Perform inference
            with autocast(device_type='cuda', enabled=True):

                mri_image = mri_image.unsqueeze(1).half().to(DEVICE)
                with torch.no_grad():
                    mri_image_encode = mri_autoencoder.encoder(mri_image)
                context_clip = process_mri_and_context(mri_image_encode, context)
                # Generate the sampled image using the diffusion process
                generated_image = sample_using_diffusion(
                    autoencoder=autoencoder,
                    diffusion=diffusion,
                    context= context_clip,
                    device=DEVICE,
                    scale_factor=scale_factor
                )
                # Assuming you want to save the first image in the batch for visualization
                image_to_save = generated_image.squeeze(0).squeeze(0)

                result= save_image_with_metadata(image_to_save, path)
                fixed_path = '/blue/kgong/gongyuxin/taupet/BrLP/scripts/result_img'
                save_path= os.path.join(fixed_path, os.path.basename(path[0]))
                print(save_path)
                sitk.WriteImage(result, save_path)
                a=1

if __name__ == '__main__':
    # Define test arguments (using the same ones from the training code)
    args = argparse.Namespace(
        image_dir="/blue/kgong/gongyuxin/taupet/ADNI_data/Latent_result2",
        mri_dir='/blue/kgong/gongyuxin/taupet/ADNI_data/MRI6',
        pet_dir='/blue/kgong/gongyuxin/taupet/ADNI_data/PET4',
        csv_path='/orange/kgong/Data/ADNI/filtered_data_PET.csv',
        cache_dir="/blue/kgong/gongyuxin/taupet/BrLP/scripts/unet_cache",
        aekl_ckpt="/blue/kgong/gongyuxin/taupet/BrLP/scripts/ae_output/autoencoder-ep-600.pth",
        aekl_ckpt_mri="/blue/kgong/gongyuxin/taupet/BrLP/scripts/ae_output_mri/autoencoder-ep-600.pth",
        diff_ckpt="/blue/kgong/gongyuxin/taupet/BrLP/scripts/unet_output/unet-ep-415.pth",
        num_workers=8,
        n_epochs=1,  # Number of epochs for testing, can be set lower for validation
        batch_size=1,
    )#121 312 348

    test(args)


