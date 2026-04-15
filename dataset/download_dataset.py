# dataset.py
import urllib.request
import tarfile
import os

ZIP_NAME = 'KuaiRand-Pure.tar.gz'
EXTRACT_DIR = './kuairand'
DOWNLOAD_URL = 'https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz'

# 1. Download if not exists
if not os.path.exists(ZIP_NAME):
    print('Downloading KuaiRand-Pure...')
    urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_NAME)
    print('Download complete.')
else:
    print(f'✔ {ZIP_NAME} already exists, skipping download.')

# 2. Extract if folder doesn't exist
if not os.path.exists(EXTRACT_DIR):
    print('Extracting...')
    with tarfile.open(ZIP_NAME, 'r:gz') as tar:
        tar.extractall(EXTRACT_DIR)
    print(f'✔ Extracted to {EXTRACT_DIR}/')
else:
    print(f'✔ {EXTRACT_DIR}/ already exists, skipping extraction.')

# 3. Show extracted files
print('\nAvailable files:')
for root, dirs, files in os.walk(EXTRACT_DIR):
    level = root.replace(EXTRACT_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    for file in files:
        size = os.path.getsize(os.path.join(root, file)) / (1024*1024)
        print(f'{indent}  {file}  ({size:.1f} MB)')