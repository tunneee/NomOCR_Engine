import gdown
import os

url = 'https://drive.usercontent.google.com/download?id=1_iyLz-nl5ezvspezRfEO16v820IZ0aeS'
output = './assets/det_model.pt'

# check if the file is already downloaded
if not os.path.exists(output):
    print('Downloading file...')
    gdown.download(url, output)
    print('Downloaded')
    
else:
    print('File already exists')
    
