import os
import glob
from torch.utils.data import Dataset
import cv2
from PIL import Image


class HistoricalColor(Dataset):
    root = r'/home/tuijiansuanfa/users/cjx/data/HistoricalColor/data/imgs/decade_database'


    def __init__(self, transform=None):
        super(Dataset, self).__init__()
        mapping = {
            '1930s': 0,
            '1940s': 1,
            '1950s': 2,
            '1960s': 3,
            '1970s': 4,
        }
        self.transform = transform
        self.targets = []
        self.images_path = []
        pattern = self.root + '/*/*.jpg'
        for image_path in glob.glob(pattern):
            self.images_path.append(image_path)
            self.targets.append(mapping[image_path.split('/')[-2].lower()])
        self.classes = list(range(5))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image = cv2.cvtColor(cv2.imread(self.images_path[item]), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        label = self.targets[item]
        return image, label


if __name__ == "__main__":
    dataset = HistoricalColorDatasets()
    print(dataset.__class__.__name__)