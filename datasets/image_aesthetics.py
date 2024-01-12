import os
import glob
from torch.utils.data import Dataset
import cv2
from PIL import Image
import pandas as pd

class ImageAesthetics(Dataset):
    root = r'/home/tuijiansuanfa/users/cjx/data/ImageAesthetics'


    def __init__(self, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.targets = []
        self.image_paths = []
        data_csv = pd.read_csv(os.path.join(self.root , 'aes.csv'))
        image_root = os.path.join(self.root , 'imgs')
        for _ , item in data_csv.iterrows():
            if int(item.iloc[2]) == 0:   continue
            self.image_paths.append(os.path.join(image_root, item.iloc[0]))
            self.targets.append(int(item.iloc[2]) - 1)

        self.classes = list(set(self.targets))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        try:
            image = cv2.cvtColor(cv2.imread(self.image_paths[item]), cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        except Exception as e:
            print(self.image_paths[item])
        if self.transform is not None:
            image = self.transform(image)
        label = self.targets[item]
        return image, label


if __name__ == "__main__":
    dataset = ImageAesthetics()
    print(dataset.__class__.__name__)