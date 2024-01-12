import os
from torch.utils.data import Dataset
import cv2
from PIL import Image
class UTKFace(Dataset):
    root = '/home/tuijiansuanfa/users/cjx/data/UTKface'
    @staticmethod
    def pre():
        intervals = [
            [0, 2], [2, 6], [6, 12], [12, 19], [19, 23], [23, 27], [27, 30],
            [30, 38], [38, 45], [45, 55], [55, 65], [65, 73], [73, 80], [80, 117]
        ]
        mapping_age_to_group = {}
        for idx, interval in enumerate(intervals):
            for x in range(interval[0], interval[1]):
                mapping_age_to_group[x] = idx
        return mapping_age_to_group

    def __init__(self, transform=None):
        super(Dataset, self).__init__()
        mapping_age_to_group = self.pre()
        # 获取所有图像名
        relative_paths = os.listdir(self.root)
        real_ages = [path.split('_')[0] for path in relative_paths]
        # 获取所有图像的绝对路径
        images_path = [os.path.join(self.root, path) for path in relative_paths]
        self.images_path = images_path
        self.transform = transform
        # 映射年龄为组标签
        self.groupID = [mapping_age_to_group[int(age)] for age in real_ages]
        self.targets = self.groupID
        #self.labels = [int(age) for age in real_ages]
        self.classes = list(set(self.targets))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image = cv2.cvtColor(cv2.imread(self.images_path[item]), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        label = self.targets[item]
        return image,label

if __name__ == "__main__":
    dataset = UTKFace()
    print(dataset.__class__.__name__)