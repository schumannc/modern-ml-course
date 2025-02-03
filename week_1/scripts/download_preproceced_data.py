import gdown
import os

# Define the Google Drive folder URL and the destination directory
url = 'https://drive.google.com/drive/u/0/folders/1BoOKkdPGPMFRzMpJL1GbnsKiz654hkzI'
destination_dir = './week_1/data/preproceced'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Download the folder contents
gdown.download_folder(url, output=destination_dir, quiet=False)
