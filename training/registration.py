import os
import nibabel as nib
import glob
import ants


def AffineReg(fixed_image_path, moving_image_path, transformed_image_path):
    fixed = ants.image_read(fixed_image_path)
    moving = ants.image_read(moving_image_path)

    # Step 2: Perform rigid registration
    registration_result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid"  # Specify rigid transformation
    )

    # Step 3: Get the transformed image and save results
    transformed_image = registration_result["warpedmovout"]  # Warped moving image
    # Save the transformed image
    # transformed_image_path = "path/to/output_transformed_image.nii"
    ants.image_write(transformed_image, transformed_image_path)
    print(f"Transformed image saved at: {transformed_image_path}")


savepath = '/blue/kgong/gongyuxin/taupet/ADNI_data/MRI4/'
mri_list = sorted(glob.glob('/blue/kgong/gongyuxin/taupet/ADNI_data/MRI3/*'))
pet_list = sorted(glob.glob('/blue/kgong/gongyuxin/taupet/ADNI_data/PET3/*'))

for i in range(len(mri_list)):
    print(mri_list[i], pet_list[i])
    fixed_image_path = pet_list[i]
    moving_image_path = mri_list[i]
    transformed_image_path = os.path.join(savepath,
                                          os.path.basename(moving_image_path).replace('.nii.gz', '_affined.nii.gz'))
    AffineReg(fixed_image_path, moving_image_path, transformed_image_path)