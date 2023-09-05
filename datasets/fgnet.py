from torch.utils.data import Dataset
from PIL import Image
import os
import re

__all__ = ["FGNET"]

class FGNET(Dataset):
    root = '/home/tuijiansuanfa/users/cjx/data/FGNET/images'

    def __init__(self, transform = None):
        super(FGNET , self).__init__()
        file_path = self.root
        file_names = os.listdir(self.root)
        img_paths = [os.path.join(file_path, file_name) for file_name in file_names]
        self.images = [Image.open(img_path).convert('RGB') for img_path in img_paths]
        self.labels = [int(re.match(r'\d{3}A(\d+)\w?.JPG', name).group(1)) for name in file_names]
        self.transform = transform
        intervals = [0, 3, 11, 16, 24, 40]
        self.groupIds = []
        for label in self.labels:
            for i in range(len(intervals) - 1, -1, -1):
                if label >= intervals[i]:
                    self.groupIds.append(i)
                    break
        self.labels = self.groupIds
        self.classes = list(set(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        #label = self.groupIds[index]
        return image, label
