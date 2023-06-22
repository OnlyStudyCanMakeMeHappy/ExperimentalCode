from torch.utils.data import Dataset
import glob
import pandas as pd
from PIL import Image

class Adience(Dataset):
    root = '/home/tuijiansuanfa/users/cjx/data/Adience'
    def __init__(self, transform=None):
        # 获取每张图像的绝对路径
        # self.img_path = os.path.join(root , 'images')
        age_mapping = [('(0, 2)', '0-2'), ('2', '0-2'), ('3', '0-2'), ('(4, 6)', '4-6'), ('(8, 12)', '8-13'),
                       ('13', '8-13'), ('22', '15-20'), ('(8, 23)', '15-20'), ('23', '25-32'), ('(15, 20)', '15-20'),
                       ('(25, 32)', '25-32'), ('(27, 32)', '25-32'), ('32', '25-32'), ('34', '25-32'), ('29', '25-32'),
                       ('(38, 42)', '38-43'), ('35', '38-43'), ('36', '38-43'), ('42', '48-53'), ('45', '38-43'),
                       ('(38, 43)', '38-43'), ('(38, 42)', '38-43'), ('(38, 48)', '48-53'), ('46', '48-53'),
                       ('(48, 53)', '48-53'), ('55', '48-53'), ('56', '48-53'), ('(60, 100)', '60+'), ('57', '60+'),
                       ('58', '60+')]
        age_mapping_dict = dict(age_mapping)
        age_to_label_map = {
            '0-2': 0,
            '4-6': 1,
            '8-13': 2,
            '15-20': 3,
            '25-32': 4,
            '38-43': 5,
            '48-53': 6,
            '60+': 7
        }
        anna_file_path = glob.glob(self.root + "/targets/*.txt")
        data_list = [pd.read_csv(path, delimiter='\t', ) for path in anna_file_path]
        data = pd.concat(data_list, ignore_index=True)
        # 删除age = None的行
        data = data.dropna(subset=['age'])
        self.labels = []
        self.image_path = []
        self.images = []
        self.transform = transform
        for _, row in data.iterrows():
            # pandas version 2.0.2
            if row.age:
                self.labels.append(age_to_label_map[age_mapping_dict[row.age]])
                img_path = f"{self.root}/images/{row.user_id}/landmark_aligned_face.{row.face_id}.{row.original_image}"
                self.image_path.append(img_path)
                #self.images.append(Image.open(img_path))
        self.classes = list(set(self.labels))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        #image = self.images[item]
        label = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
