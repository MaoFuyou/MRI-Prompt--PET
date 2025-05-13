
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


def fill_mmse_scores(data_pet_path, data_mri_path, output_pet_path, output_mri_path):
    """
    Fill in missing MMSE Total Scores in data_PET and data_MRI tables.

    Parameters:
        data_pet_path (str): File path for the data_PET table.
        data_mri_path (str): File path for the data_MRI table.
        output_pet_path (str): Output path for the completed data_PET table.
        output_mri_path (str): Output path for the completed data_MRI table.

    Returns:
        None
    """
    # Load the data
    data_PET = pd.read_csv(data_pet_path)
    data_MRI = pd.read_csv(data_mri_path)

    # Ensure Study Date is in datetime format
    data_PET['Study Date'] = pd.to_datetime(data_PET['Study Date'], errors='coerce', format='%m/%d/%Y')
    data_MRI['Study Date'] = pd.to_datetime(data_MRI['Study Date'], errors='coerce', format='%m/%d/%Y')

    # Iterate over data_PET to fill missing MMSE Total Scores
    for i, row in data_PET.iterrows():
        if pd.isna(row['MMSE Total Score']):  # If MMSE Total Score is missing
            subject_id = row['Subject ID']
            pet_date = row['Study Date']

            # Filter data_MRI rows with the same Subject ID
            matching_rows = data_MRI[data_MRI['Subject ID'] == subject_id].copy()

            # If there are matching rows
            if not matching_rows.empty:
                # Calculate the date difference between Study Date and pet_date
                matching_rows['Date Diff'] = abs(matching_rows['Study Date'] - pet_date)

                # Filter rows where MMSE Total Score is not missing
                matching_rows_with_mmse = matching_rows[~matching_rows['MMSE Total Score'].isna()]

                # If there are rows with a non-missing MMSE Total Score
                if not matching_rows_with_mmse.empty:
                    # Find the row with the closest Study Date
                    closest_row = matching_rows_with_mmse.sort_values('Date Diff').iloc[0]
                    # Use this row's MMSE Total Score to fill in the missing value in data_PET
                    data_PET.at[i, 'MMSE Total Score'] = closest_row['MMSE Total Score']

    # Iterate over data_MRI to fill missing MMSE Total Scores
    for i, row in data_MRI.iterrows():
        if pd.isna(row['MMSE Total Score']):
            # If MMSE Total Score is missing
            subject_id = row['Subject ID']

            mri_date = row['Study Date']

            # Filter data_PET rows with the same Subject ID
            matching_rows = data_PET[data_PET['Subject ID'] == subject_id].copy()

            # If there are matching rows
            if not matching_rows.empty:
                # Calculate the date difference between Study Date and mri_date
                matching_rows['Date Diff'] = abs(matching_rows['Study Date'] - mri_date)

                # Filter rows where MMSE Total Score is not missing
                matching_rows_with_mmse = matching_rows[~matching_rows['MMSE Total Score'].isna()]

                # If there are rows with a non-missing MMSE Total Score
                if not matching_rows_with_mmse.empty:
                    # Find the row with the closest Study Date
                    closest_row = matching_rows_with_mmse.sort_values('Date Diff').iloc[0]
                    # Use this row's MMSE Total Score to fill in the missing value in data_MRI
                    data_MRI.at[i, 'MMSE Total Score'] = closest_row['MMSE Total Score']

    # Save the updated tables to new Excel files
    data_PET.to_csv(output_pet_path, index=False)
    data_MRI.to_csv(output_mri_path, index=False)
    print("MMSE Total Score completion is complete!")
    return data_PET,data_MRI


def process_pet_data(data_PET, base_path, output_csv, output_folder):
    """
    Process PET data to filter and convert images.

    Args:
        data_PET (pd.DataFrame): The input DataFrame containing PET data.
        base_path (str): The base path for the PET data.
        output_csv (str): The path to save the filtered CSV data.
        output_folder (str): The output folder to save converted PET images.
    """
    filtered_data = []

    for idx, row in data_PET.iterrows():
        # Extract necessary fields
        subject_id = row['Subject ID']
        description = row['Description']
        study_date = row['Study Date'].date()
        image_id = row['Image ID']
        MMSE = row['MMSE Total Score']
        diagnosis = row['Research Group']
        Sex = row['Sex']
        Age = row['Age']


        # Skip rows with missing MMSE
        if pd.isna(MMSE):
            continue

        # Format study date
        try:
            study_date = study_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error processing study date for Subject ID {subject_id}: {e}")
            continue

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
                output_filename = f"{subject_id}_{MMSE}_{study_date}_PET.nii.gz"
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
                        'MMSE': MMSE
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


def process_mri_data(data_MRI, base_path, output_csv, output_folder):
    """
    Process PET data to filter and convert images.

    Args:
        data_PET (pd.DataFrame): The input DataFrame containing PET data.
        base_path (str): The base path for the PET data.
        output_csv (str): The path to save the filtered CSV data.
        output_folder (str): The output folder to save converted PET images.
    """
    filtered_data = []

    for idx, row in data_MRI.iterrows():
        # Extract necessary fields
        subject_id = row['Subject ID']
        description = row['Description']
        study_date = row['Study Date'].date()
        image_id = row['Image ID']
        MMSE = row['MMSE Total Score']
        diagnosis = row['Research Group']
        Sex = row['Sex']
        Age = row['Age']

        # Skip rows with missing MMSE
        if pd.isna(MMSE):
            continue

        # Format study date
        try:
            study_date = study_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error processing study date for Subject ID {subject_id}: {e}")
            continue

        # Prepare paths and descriptions
        description_folder = description.split('<')[0].strip().replace("; ", "__")
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
                    output_filename = f"{subject_id}_{MMSE}_{study_date}_{matching_id[i][0]}_MRI.nii.gz"
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
                            'MMSE': MMSE
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

    # data = pd.read_csv("/orange/kgong/Data/ADNI/data.csv")
    # # data['Description'].str.startswith('AV45 Coreg, Avg, Standardized Image and Voxel Size') |
    # filtered_data = data[
    #     (data['Modality'] == 'PET') &
    #     (
    #         (data['Description'].str.startswith('AV1451 Coreg, Avg, Standardized Image and Voxel Size') & data[
    #             'Description'].str.endswith('Tau'))
    #     )
    #     ]
    #
    # filtered_data.to_csv("/orange/kgong/Data/ADNI/data_PET.csv", index=False)
    #
    # filtered_data = data[((data['Modality'] == 'MRI') &
    #                       (data['Description'].str.startswith('MT1; N3m '))) |
    #                      (data['Description'] == 'Accelerated Sagittal MPRAGE')]
    # filtered_data.to_csv("/orange/kgong/Data/ADNI/data_MRI.csv", index=False)
    #
    # ######
    data_PET, data_MRI = fill_mmse_scores(
        data_pet_path='/orange/kgong/Data/ADNI/data_PET.csv',
        data_mri_path='/orange/kgong/Data/ADNI/data_MRI.csv',
        output_pet_path='/orange/kgong/Data/ADNI/data_PET2.csv',
        output_mri_path='/orange/kgong/Data/ADNI/data_MRI2.csv'
    )

    ####
    base_path = '/orange/kgong/Data/ADNI/download/ADNI'
    output_csv = "/orange/kgong/Data/ADNI/filtered_data_PET.csv"
    output_folder = '/blue/kgong/gongyuxin/taupet/ADNI_data/pet2'

    filtered_df = process_pet_data(data_PET, base_path, output_csv, output_folder)
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data saved successfully to {output_csv}!")

    ####
    base_path = '/orange/kgong/Data/ADNI/download/ADNI'
    output_csv = "/orange/kgong/Data/ADNI/filtered_data_MRI.csv"
    output_folder = '/blue/kgong/gongyuxin/taupet/ADNI_data/mri2'

    filtered_df = process_mri_data(data_MRI, base_path, output_csv, output_folder)
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data saved successfully to {output_csv}!")










