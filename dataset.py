from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

    
class expand_greyscale(object):
    def __init__(self):
        self.num_target_channels = 3

    def __call__(self, tensor):
        channels = tensor.shape[0]
        
        if channels == self.num_target_channels:
            return tensor
        elif channels == 1:
            color = tensor.expand(3, -1, -1)
            return color
            

    
    
class XRayDataset(Dataset):
    def __init__(self, path, resolution=256):
        super().__init__()
        self.folder = path #csv_file_name
        import pandas as pd
        df = pd.read_csv(path)
        self.paths = list(np.asarray(df['lateral_512_jpeg']))
        print("Number of files: ", len(self.paths))
        self.length = len(self.paths)

        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale())#,
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        #img = np.asarray(img)
        #img = img / 255.0 
        #img = img - 0.5
        #img = img * 2.0
        img = self.transform(img)
        return img
