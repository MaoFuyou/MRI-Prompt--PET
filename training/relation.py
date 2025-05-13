import os
import numpy as np
import nibabel as nib
import re
import matplotlib.pyplot as plt

# ¾ö9ï
mri_dir = "/blue/kgong/gongyuxin/taupet/ADNI_data/mri4"
pet_dir = "/blue/kgong/gongyuxin/taupet/ADNI_data/pet3"
mask_dir = "/blue/kgong/gongyuxin/taupet/ADNI_data/entorinal_region_result3"


mri_files = [f for f in os.listdir(mri_dir) if f.endswith(".nii.gz")]
matched_data = {}
left_avg_list = []
right_avg_list = []
value_list = []

for mri_file in mri_files:
    match = re.match(r"(.*)_MRI_affined.nii.gz", mri_file)
    if match:
        base_name = match.group(1)
        pet_file = f"{base_name}_PET.nii.gz"
        mask_folder = os.path.join(mask_dir, f"{base_name}_MRI_affined/mri")

        if os.path.exists(os.path.join(pet_dir, pet_file)) and os.path.exists(mask_folder):
            matched_data[base_name] = {
                "mri_path": os.path.join(mri_dir, mri_file),
                "pet_path": os.path.join(pet_dir, pet_file),
                "left_mask": os.path.join(mask_folder, "entorhinal_left_mask_resampled.nii.gz"),
                "right_mask": os.path.join(mask_folder, "entorhinal_right_mask_resampled.nii.gz")
            }

for key, paths in matched_data.items():
    pet_img = nib.load(paths["pet_path"]).get_fdata()
    left_mask_img = nib.load(paths["left_mask"]).get_fdata()
    right_mask_img = nib.load(paths["right_mask"]).get_fdata()

    left_masked = pet_img * left_mask_img
    right_masked = pet_img * right_mask_img

    left_avg = np.mean(left_masked[left_masked > 0])
    right_avg = np.mean(right_masked[right_masked > 0])

    match = re.search(r"_(\d+\.\d+)_", key)
    value = float(match.group(1)) if match else np.nan

    left_avg_list.append(left_avg)
    right_avg_list.append(right_avg)
    value_list.append(value)

    print(f"{key}: Left Mask Avg = {left_avg:.2f}, Right Mask Avg = {right_avg:.2f}, Extracted Value = {value}")

correlation_left = np.corrcoef(left_avg_list, value_list)[0, 1]
correlation_right = np.corrcoef(right_avg_list, value_list)[0, 1]


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(left_avg_list, value_list, color='blue', alpha=0.7, edgecolors='black',
            label=f'Corr: {correlation_left:.2f}')
plt.xlabel("Left Mask Avg")
plt.ylabel("Plasma Value")
plt.title("Scatter Plot: Left Mask Avg vs Plasma Value")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(right_avg_list, value_list, color='red', alpha=0.7, edgecolors='black',
            label=f'Corr: {correlation_right:.2f}')
plt.xlabel("Right Mask Avg")
plt.ylabel("Plasma Value")
plt.title("Scatter Plot: Right Mask Avg vs Plasma Value")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()


plt.tight_layout()
plt.show()


print(f"Left Mask Avg vs Value øs'ûp: {correlation_left:.2f}")
print(f"Right Mask Avg vs Value øs'ûp: {correlation_right:.2f}")
