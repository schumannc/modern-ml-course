import gdown
import os

# Define the Google Drive folder URL and the destination directory
url = 'https://drive.google.com/drive/folders/1NVQtamDRuGuAIZCRfidvnlSPH1-xLGlG'
destination_dir = './case/data/processed'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Download the folder contents
gdown.download_folder(url, output=destination_dir, quiet=False)
