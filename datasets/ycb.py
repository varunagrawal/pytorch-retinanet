import os

import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageLoaderAll(Dataset):

    def __init__(self, root, train_file_path, transform=None, flip=False, affine=False, train=False,
                 batch_size=64, num_workers=4):

        super().__init__()

        self.root = root
        self.train_list = get_data_list(train_file_path)
        self.num_samples = len(self.train_list)
        self.transform = transform
        self.flip = flip
        self.affine = affine
        self.train = train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     "data", "LOV", "data")

    def get_data(self, dir_path, transform, flip, affine, root, random=None, key=False):
        meta_path = os.path.join(dir_path + "-meta.mat")
        bbox_path = os.path.join(dir_path + "-box.txt")
        img_path = os.path.join(dir_path + "-color.png")
        label_path = os.path.join(dir_path + '-label.png')
        depth_path = os.path.join(dir_path + '-depth.png')

        classes = get_classes(root)
        meta_data = sio.loadmat(meta_path)
        bbox = []
        poses = []
        cls_indices = []

        img = Image.open(img_path)
        # img = img.resize((int(img.width/10), int(img.height/10)), Image.ANTIALIAS)
        label = np.array(Image.open(label_path))
        depth = np.array(Image.open(depth_path)) / meta_data["factor_depth"]

        # label single channel to one hot
        labels = np.zeros((22, label.shape[0], label.shape[1]), dtype=float)
        for j in range(0, 22):
            labels[j, :, :] = ((label == (j)) * 1)

        with open(bbox_path, 'r') as file:
            lines = file.readlines()

        # Randomly selects an object
        if random is None:
            random = np.random.choice(len(lines), len(lines), replace=False)

        for i in random:
            lis = str(lines[i]).split("\n")[0].split(" ")
            if float(lis[1]) <= 640.0 and float(lis[3]) <= 640 \
                and float(lis[2]) <= 480 and float(lis[4]) <= 480 \
                    and (float(lis[3]) > float(lis[1])) and (float(lis[4]) > float(lis[2])) \
                        and (float(lis[3]) - float(lis[1])) > 20 \
                            and (float(lis[4]) - float(lis[2])) > 20:
                bbox.append([float(lis[1]), float(lis[2]),
                             float(lis[3]), float(lis[4])])
                cls_indices.append(classes[lis[0]])

        for i in cls_indices:
            ind = list(meta_data['cls_indexes']).index(i)
            poses.append(meta_data['poses'][:, :, ind])

        if transform:
            img = transform(img)

        return img, labels, depth, poses, cls_indices, bbox, 0, meta_data['intrinsic_matrix'],\
            meta_data['rotation_translation_matrix'], 0, random

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        # train_list has key from 0001 to 0090, num videos
        data = {"image": [], "poses": [], "cls_indices": [], "bbox": [],
                "file_indices": [], "K": [], "RT": [], "label": [], "depth": []}

        file_indices = self.train_list[index]
        img, label, depth, poses, cls_indices, bbox, _, K, RT, _, _ = self.get_data(
            os.path.join(self.data_dir, file_indices),
            self.transform, self.flip, self.affine, self.root)

        data['image'] = img
        data['label'] = label
        data['depth'] = depth
        data['poses'] = poses
        data['cls_indices'] = cls_indices
        data['bbox'] = bbox
        data['file_indices'] = file_indices
        data['K'] = K
        data['RT'] = RT

        return data


def gaussian_kernel(size_x, size_y, ux=0, uy=0, a=0, b=0):
    x, y = np.mgrid[0:size_x, 0:size_y]
    g = np.exp(-(((x-ux)/a)**2 + ((y-uy)/b)**2))
    return g


def get_crops(img, bbox):
    im = []
    for i in range(0, len(bbox)):
        x1, x2, y1, y2 = int(bbox[i][0]), int(
            bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])
        crop = Image.Image.resize(Image.Image.crop(
            img, [x1, y1, x2, y2]), (64, 64))
        im.append(transforms.ToTensor()(crop))
    return im


def get_matrix(deg, nx, ny):
    a = np.math.cos(deg * np.pi / 180)
    b = np.math.sin(deg * np.pi / 180)
    c = nx + nx*a - ny*b
    d = nx + nx*-b - ny*a
    mat = [[a, b, c], [-b, a, d], [0, 0, 1]]
    return mat, a, b, c, d


def get_data_list(root):
    train_list = []

    with open(root, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines)):
        lis = lines[i].split("\n")[0]
        train_list.append(lis)

    return train_list


def get_classes(root):
    classes_file = os.path.join(root, 'classes.txt')
    classes = {}
    with open(classes_file, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines)):
        classes[str(lines[i]).split("\n")[0]] = i + 1

    return classes


def load_object_points_dense(root):
    classes = get_classes(root)
    points = [[] for _ in range(0, len(classes))]
    num = np.inf

    for i in range(0, len(classes)):
        point_file = os.path.join(root, 'models',
                                  list(classes.keys())[
                                      list(classes.values()).index(i+1)],
                                  'points1.xyz')
        assert os.path.exists(
            point_file), 'Path does not exist: {}'.format(point_file)
        points[i] = np.loadtxt(point_file)
        if points[i].shape[0] < num:
            num = points[i].shape[0]

    new_num = 3000
    points_all = np.zeros((len(classes), new_num, 3), dtype=np.float32)
    for i in range(0, len(classes)):
        indices = np.random.randint(0, points[i].shape[0], new_num)
        points_all[i, :, :] = points[i][indices, :]

    return points, points_all


def load_object_points(root):
    classes = get_classes(root)
    points = [[] for _ in range(0, len(classes))]
    num = np.inf

    for i in range(0, len(classes)):
        point_file = os.path.join(root, 'models',
                                  list(classes.keys())[
                                      list(classes.values()).index(i + 1)],
                                  'points.xyz')
        assert os.path.exists(
            point_file), 'Path does not exist: {}'.format(point_file)
        points[i] = np.loadtxt(point_file)
        if points[i].shape[0] < num:
            num = points[i].shape[0]

    points_all = np.zeros((len(classes), num, 3), dtype=np.float32)
    for i in range(0, len(classes)):
        points_all[i, :, :] = points[i][:num, :]

    return points, points_all


def main():
    file = "/home/varun/projects/VideoPose/VideoPose/data/LOV/train_video.txt"
    shuffled = open("/home/varun/projects/VideoPose/VideoPose/data/LOV/train_video_shuffled.txt",
                    'w')
    train_list = get_data_list(file)


if __name__ == '__main__':
    main()
