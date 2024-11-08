import gdown
import os

# url = 'https://drive.google.com/uc?id=157gqq8aoNmrj9B35-sxeaFStsmDe891H'
# output = './assets/det_model.zip'

url = 'https://drive.google.com/uc?id=1Ok3gNQa_7RcBBvUakDIYMdbaW6POMagK'
output = './assets/det_model.pt'

# check if the file is already downloaded
if not os.path.exists(output):
    print('Downloading file...')
    gdown.download(url, output)
    print('Downloaded')
    
else:
    print('File already exists')
    
