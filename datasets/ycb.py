import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from bbox import BBox2D


class YCBDataset(Dataset):

    def __init__(self, root, data_file_path, transform=None, train=False):

        super().__init__()

        self.root = root
        self.data_list = self.get_data_list(osp.join(root, data_file_path))
        self.num_samples = len(self.data_list)
        self.transform = transform
        self.train = train
        self.data_dir = osp.join(root, "data")
        self.classes = self.get_classes(self.root)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    @staticmethod
    def get_data_list(root):
        data_list = []

        with open(root, 'r') as file:
            lines = file.readlines()

        for line in lines:
            lis = line.split("\n")[0]
            data_list.append(lis)

        return data_list

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        img = self.load_image(image_index)
        return float(img.shape[1]) / float(img.shape[0])

    @staticmethod
    def get_classes(root):
        classes_file = osp.join(root, 'image_sets', 'classes.txt')
        classes = {}
        with open(classes_file, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            classes[str(line).split("\n")[0]] = i

        return classes

    def num_classes(self):
        return len(self.classes)

    def load_image(self, index):
        # data_list has key from 0001 to 0090, num videos
        file_indices = self.data_list[index]
        dir_path = osp.join(self.data_dir, file_indices)
        img_path = osp.join(dir_path + "-color.png")

        img = Image.open(img_path)
        # img = img.resize((int(img.width/10), int(img.height/10)), Image.ANTIALIAS)

        return np.asfarray(img) / 255.0

    def load_annotations(self, index):
        # data_list has key from 0001 to 0090, num videos
        file_indices = self.data_list[index]
        dir_path = osp.join(self.data_dir, file_indices)

        bbox_path = osp.join(dir_path + "-box.txt")

        bboxes = []
        cls_indices = []

        with open(bbox_path, 'r') as file:
            lines = file.readlines()

        random = np.random.choice(len(lines), len(lines), replace=False)

        for i in random:
            lis = str(lines[i]).split("\n")[0].split(" ")
            box = BBox2D(list(map(float, lis[1:])), mode=1)
            if box.x1 <= 640.0 and box.x2 <= 640 and box.y1 <= 480 and box.y2 <= 480 \
                    and (box.x2 > box.x1) and (box.y2 > box.y1) \
                and box.w > 20 and box.h > 20:
                bboxes.append(box.tolist(1))
                cls_indices.append(self.classes[lis[0]])

        annot = np.zeros((len(bboxes), 5))
        annot[:, :4] = np.array(bboxes)
        annot[:, 4] = cls_indices

        return annot

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img = self.load_image(index)
        annot = self.load_annotations(index)

        sample = {'img': np.array(img), 'annot': annot}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_crops(img, bbox):
    im = []
    for i in range(0, len(bbox)):
        x1, x2, y1, y2 = int(bbox[i][0]), int(
            bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])
        crop = Image.Image.resize(Image.Image.crop(
            img, [x1, y1, x2, y2]), (64, 64))
        im.append(transforms.ToTensor()(crop))
    return im


def main():
    root = "/home/varun/projects/VideoPose/VideoPose/data/LOV/"
    file = "train_video.txt"

    dataset = YCBDataset(root, file, None, train=True)
    dataset[0]


if __name__ == '__main__':
    main()
