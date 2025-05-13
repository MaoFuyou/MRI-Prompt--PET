
import SimpleITK as sitk
import os
import pandas as pd
from datetime import datetime


def convert_dcm_to_nii(input_folder, output_folder, output_filename):
    try:
        os.makedirs(output_folder, exist_ok=True)

        files = os.listdir(input_folder)
        dcm_files = [f for f in files if f.endswith(".dcm")]
        nii_files = [f for f in files if f.endswith(".nii")]

        if dcm_files:
            print("Detected DICOM files, converting to NIfTI...")
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(input_folder)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

        elif nii_files:
            print("Detected NIfTI file, converting to .nii.gz...")
            nii_file_path = os.path.join(input_folder, nii_files[0])
            image = sitk.ReadImage(nii_file_path)

        else:
            print("No valid DICOM or NIfTI files found in the input folder.")
            return

        output_path = os.path.join(output_folder, output_filename)

        sitk.WriteImage(image, output_path)

        print(f"Successfully transformed and saved: {output_path}")

    except Exception as e:
        print(f"Transformation failed: {e}")

def process_pet_data(data_PET, base_path, output_csv, output_folder,plasma_folder,plasma_folder2):
    """
    Process PET data to filter and convert images.

    Args:
        data_PET (pd.DataFrame): The input DataFrame containing PET data.
        base_path (str): The base path for the PET data.
        output_csv (str): The path to save the filtered CSV data.
        output_folder (str): The output folder to save converted PET images.
    """
    filtered_data = []
    PET_plasma1 = pd.read_csv(plasma_folder)
    PET_plasma2 = pd.read_csv(plasma_folder2)

    for idx, row in data_PET.iterrows():
        # Extract necessary fields
        subject_id = row['Subject']
        description = row['Description']
        study_date = row['Acq Date']#.date()
        image_id = row['Image Data ID']
        diagnosis = row['Group']
        Sex = row['Sex']
        Age = row['Age']

        matched_row1 = PET_plasma1[PET_plasma1['PTID'] == subject_id]
        matched_row2 = PET_plasma2[PET_plasma2['PTID'] == subject_id]

        if matched_row1.empty and matched_row2.empty:
            continue
        if 'TESTVALUE' not in matched_row1.columns and 'TESTVALUE' not in matched_row2.columns:
            continue
        if not matched_row1.empty:
            test_value = matched_row1['TESTVALUE'].values[0]
        else:
            test_value = matched_row2['TESTVALUE'].values[0]

        from datetime import datetime

        try:
            study_date = datetime.strptime(study_date, '%m/%d/%Y')  # Convert string to datetime object
            study_date = study_date.strftime('%Y-%m-%d')  # Format datetime as 'YYYY-MM-DD' string
            print("Formatted study date:", study_date)
        except Exception as e:
            print(f"Error processing study date: {e}")

        # Prepare paths and descriptions
        description_folder = description.split('<')[0].strip().replace(" ", "_")
        base_path2 = os.path.join(base_path, subject_id, description_folder)

        # Find matching folders
        matching_folders = []
        matching_id = []

        if os.path.exists(base_path2):
            for folder in os.listdir(base_path2):
                folder_path = os.path.join(base_path2, folder)
                contents = os.listdir(folder_path)
                subfolders = [name for name in contents if os.path.isdir(os.path.join(folder_path, name))]
                if os.path.isdir(folder_path) and study_date in folder:
                    matching_folders.append(folder)
                    matching_id.append(subfolders)

        # Process matching folders
        if matching_folders:
            try:
                image_folder = os.path.join(base_path2, matching_folders[-1], f"{matching_id[-1][0]}")
                output_filename = f"{subject_id}_{test_value}_{study_date}_PET.nii.gz"#MMSE
                convert_dcm_to_nii(image_folder, output_folder, output_filename)

                image_folder2 = os.path.join(output_folder, output_filename)
                if os.path.exists(image_folder2):
                    filtered_data.append({
                        'subject_id': subject_id,
                        'image_uid': image_id,
                        'Sex ': Sex,
                        'Age ': Age,
                        'diagnosis': diagnosis,
                        'image_path': image_folder2,
                        'Description': description,
                        'Study Date': study_date,
                        "TESTVALUE": test_value

                    })
                else:
                    print(f"Warning: Converted image not found for Subject ID {subject_id}, Image ID {image_id}")
            except Exception as e:
                print(f"Error processing matching folder for Subject ID {subject_id}: {e}")
        else:
            print(f"Warning: No matching folder found for Subject ID {subject_id}, Study Date {study_date}")

    # Save filtered data to CSV
    filtered_df = pd.DataFrame(filtered_data)

    return filtered_df



def process_mri_data(data_MRI, base_path, output_csv, output_folder,plasma_folder,plasma_folder2):
    """
    Process PET data to filter and convert images.

    Args:
        data_PET (pd.DataFrame): The input DataFrame containing PET data.
        base_path (str): The base path for the PET data.
        output_csv (str): The path to save the filtered CSV data.
        output_folder (str): The output folder to save converted PET images.
    """
    filtered_data = []
    MRI_plasma1 = pd.read_csv(plasma_folder)
    MRI_plasma2 = pd.read_csv(plasma_folder2)

    for idx, row in data_MRI.iterrows():
        # Extract necessary fields
        subject_id = row['Subject']
        description = row['Description']
        study_date = row['Acq Date']
        image_id = row['Image Data ID']
        diagnosis = row['Group']
        Sex = row['Sex']
        Age = row['Age']

        matched_row1 = MRI_plasma1[MRI_plasma1['PTID'] == subject_id]
        matched_row2 = MRI_plasma2[MRI_plasma2['PTID'] == subject_id]

        if matched_row1.empty and matched_row2.empty:
            continue
        if 'TESTVALUE' not in matched_row1.columns and 'TESTVALUE' not in matched_row2.columns:
            continue
        if not matched_row1.empty:
            test_value = matched_row1['TESTVALUE'].values[0]
        else:
            test_value = matched_row2['TESTVALUE'].values[0]

        from datetime import datetime

        try:
            study_date = datetime.strptime(study_date, '%m/%d/%Y')  # Convert string to datetime object
            study_date = study_date.strftime('%Y-%m-%d')  # Format datetime as 'YYYY-MM-DD' string
            print("Formatted study date:", study_date)
        except Exception as e:
            print(f"Error processing study date: {e}")
        # Prepare paths and descriptions
        description_folder  = description.split('<')[0].strip().replace("; ", "__").replace(" ", "_")

        base_path2 = os.path.join(base_path, subject_id, description_folder)

        # Find matching folders
        matching_folders = []
        matching_id = []

        if os.path.exists(base_path2):
            for folder in os.listdir(base_path2):
                folder_path = os.path.join(base_path2, folder)
                contents = os.listdir(folder_path)
                subfolders = [name for name in contents if os.path.isdir(os.path.join(folder_path, name))]
                if os.path.isdir(folder_path) and study_date in folder:
                    matching_folders.append(folder)
                    matching_id.append(subfolders)

        # Process matching folders
        if matching_folders:
            try:
                for i in range(len(matching_folders)):
                    image_folder = os.path.join(base_path2, matching_folders[i], f"{matching_id[i][0]}")
                    output_filename = f"{subject_id}_{test_value}_{study_date}_{matching_id[i][0]}_MRI.nii.gz"
                    convert_dcm_to_nii(image_folder, output_folder, output_filename)

                    image_folder2 = os.path.join(output_folder, output_filename)
                    if os.path.exists(image_folder2):
                        filtered_data.append({
                            'subject_id': subject_id,
                            'image_uid': image_id,
                            'Sex ': Sex,
                            'Age ': Age,
                            'diagnosis': diagnosis,
                            'image_path': image_folder2,
                            'Description': description,
                            'Study Date': study_date,
                            "TESTVALUE": test_value
                        })
                else:
                    print(f"Warning: Converted image not found for Subject ID {subject_id}, Image ID {image_id}")
            except Exception as e:
                print(f"Error processing matching folder for Subject ID {subject_id}: {e}")
        else:
            print(f"Warning: No matching folder found for Subject ID {subject_id}, Study Date {study_date}")

    # Save filtered data to CSV
    filtered_df = pd.DataFrame(filtered_data)

    return filtered_df

if __name__ == '__main__':

    data = pd.read_csv("/blue/kgong/gongyuxin/taupet/ADNI_data/adni_1234go_av1451_tau_T1_3_28_2025.csv")
    # data['Description'].str.startswith('AV45 Coreg, Avg, Standardized Image and Voxel Size') |
    filtered_data = data[
        (data['Modality'] == 'PET') &
        (
            (data['Description'].str.startswith('AV1451 Coreg, Avg, Standardized Image and Voxel Size') )
        )
        ]

    filtered_data.to_csv("/blue/kgong/gongyuxin/taupet/ADNI_data/data_PET.csv", index=False)

    filtered_data = data[
        (data['Modality'] == 'MRI') & (
                data['Description'].str.startswith('MPR; GradWarp; B1 Correction; N3; Scaled') |
                (data['Description'] == 'Spatially Normalized, Masked and N3 corrected T1 image') |
                (data['Description'] == 'Accelerated Sagittal MPRAGE')
        )
        ]

    filtered_data.to_csv("/blue/kgong/gongyuxin/taupet/ADNI_data/data_MRI.csv", index=False)

    ####
    base_path = '/blue/kgong/shared_ziqian/ADNI_1234go_av1451_T1/ADNI'
    output_csv = "/blue/kgong/gongyuxin/taupet/ADNI_data/filtered_data_PET.csv"
    output_folder = '/blue/kgong/gongyuxin/taupet/ADNI_data/PET2'
    plasma_folder= "/blue/kgong/gongyuxin/taupet/ADNI_data/Taupet_plasma.csv"
    plasma_folder2='/blue/kgong/gongyuxin/taupet/ADNI_data/C2N_PRECIVITYAD2_PLASMA_26Feb2025.csv'

    data_PET = pd.read_csv("/blue/kgong/gongyuxin/taupet/ADNI_data/data_PET.csv", low_memory=False)


    # filtered_df = process_pet_data(data_PET, base_path, output_csv, output_folder, plasma_folder,plasma_folder2)
    # filtered_df.to_csv(output_csv, index=False)
    # print(f"Filtered data saved successfully to {output_csv}!")

    ####
    base_path =  '/blue/kgong/shared_ziqian/ADNI_1234go_av1451_T1/ADNI'
    output_csv = "/blue/kgong/gongyuxin/taupet/ADNI_data/filtered_data_MRI.csv"
    output_folder = '/blue/kgong/gongyuxin/taupet/ADNI_data/MRI2'
    plasma_folder = "/blue/kgong/gongyuxin/taupet/ADNI_data/Taupet_plasma.csv"
    plasma_folder2 = '/blue/kgong/gongyuxin/taupet/ADNI_data/C2N_PRECIVITYAD2_PLASMA_26Feb2025.csv'

    data_MRI = pd.read_csv("/blue/kgong/gongyuxin/taupet/ADNI_data/data_MRI.csv", low_memory=False)

    filtered_df = process_mri_data(data_MRI, base_path, output_csv, output_folder, plasma_folder,plasma_folder2)
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data saved successfully to {output_csv}!")










