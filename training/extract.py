import os
from datetime import datetime
import shutil

# Function to extract the date from the filename
def extract_date(filename):
    parts = filename.split("_")
    date_str = parts[4]  # Date part
    return datetime.strptime(date_str, "%Y-%m-%d")  # Ensure it returns a datetime object

# Function to extract the ID from the filename (e.g., 002_S_0295_22.0)
def extract_id(filename):
    return "_".join(filename.split("_")[:4])

# Define paths
pet2_dir = "/blue/kgong/gongyuxin/taupet/ADNI_data/PET2"
mri2_dir = "/blue/kgong/gongyuxin/taupet/ADNI_data/MRI2"
pet3_dir = "/blue/kgong/gongyuxin/taupet/ADNI_data/PET3"
mri3_dir = "/blue/kgong/gongyuxin/taupet/ADNI_data/MRI3"

# Create output directories
os.makedirs(pet3_dir, exist_ok=True)
os.makedirs(mri3_dir, exist_ok=True)

# Get the list of PET and MRI files
pet_files = [f for f in os.listdir(pet2_dir) if f.endswith(".nii.gz")]
mri_files = [f for f in os.listdir(mri2_dir) if f.endswith(".nii.gz")]

# Process each PET file
for pet_file in pet_files:
    pet_id = extract_id(pet_file)
    pet_date = extract_date(pet_file)

    # Find MRI files that match the PET ID
    matched_mri_files = [f for f in mri_files if pet_id in f]

    if not matched_mri_files:
        # Skip if no matching MRI files are found
        continue

    # Calculate the date difference for each MRI file
    closest_mri_file = None
    min_date_diff = float("inf")
    same_date_files = []  # List to store files with the same date difference

    for mri_file in matched_mri_files:
        mri_date = extract_date(mri_file)
        date_diff = abs((pet_date - mri_date).days)

        if date_diff < min_date_diff:
            # If the current MRI file has a smaller date difference, update the closest file
            closest_mri_file = mri_file
            min_date_diff = date_diff
            same_date_files = [mri_file]  # Start a new list of files with the same date difference
        elif date_diff == min_date_diff:
            # If the date difference is the same, add the file to the list
            same_date_files.append(mri_file)

    # If there are multiple files with the same date difference, choose the last one
    if len(same_date_files) > 1:
        closest_mri_file = same_date_files[-1]

    if closest_mri_file:
        # Construct new filenames
        new_pet_name = pet_file
        new_mri_name = new_pet_name.replace("_PET", "_MRI")

        # Copy files to the target directories
        shutil.copy(os.path.join(pet2_dir, pet_file), os.path.join(pet3_dir, new_pet_name))
        shutil.copy(os.path.join(mri2_dir, closest_mri_file), os.path.join(mri3_dir, new_mri_name))

        print(f"Processed: {pet_file} -> {new_pet_name}")
        print(f"Processed: {closest_mri_file} -> {new_mri_name}")
