import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_kaggle_dataset(competition, path):
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition, path=path)
    
    # Unzip all files in the directory
    for file_name in os.listdir(path):
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(path, file_name), 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(os.path.join(path, file_name))

if __name__ == "__main__":
    dataset_identifier = 'rossmann-store-sales'
    download_path = os.path.join('input_data', 'rossmann-store-sales')
    os.makedirs(download_path, exist_ok=True)

    download_kaggle_dataset(dataset_identifier, download_path)
    print(f"Dataset downloaded and extracted to {download_path}")
