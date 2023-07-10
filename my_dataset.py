import numpy as np
import os
import time
import cv2
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class GenerateHeatmap(object):
    def __init__(self, output_res, num_parts, sigma=3):
        self.output_res = output_res
        self.num_parts = num_parts

        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0:
                x, y = int(pt[0]), int(pt[1])
                print(x, y)
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class ToothDataSet(Dataset):
    def __init__(self, img_dir, label_txt_file, img_size=256, out_size=64, n_kps=3):
        self.img_dir = img_dir
        self.label_txt_file = label_txt_file            # img_name, x1, y1, x2, y2...
        self.img_size = img_size
        self.out_size = out_size
        self.ratio = out_size / img_size
        self.n_kps = n_kps
        self.gen_heat_map = GenerateHeatmap(self.out_size, self.n_kps)

        self.img_names = []
        self.keypoints = []
        self.visibles = []
        self.load_data()

    def load_data(self):
        print('loading data...')
        tic = time.time()

        labels = np.loadtxt(self.label_txt_file,
                            dtype={'names': ('img_name', 'left_x', 'left_y', 'mid_x', 'mid_y', 'right_x', 'right_y'),
                                   'formats': ('|S200', float, float, float, float, float, float)},
                            delimiter=';', skiprows=0)

        img_names = []
        kps = []
        visibles = []
        for i in range(len(labels)):
            img_names.append(str(labels['img_name'][i].decode("utf8")))
            kps.append([[labels['left_x'][i], labels['left_y'][i]], [labels['mid_x'][i], labels['mid_y'][i]], [labels['right_x'][i], labels['right_y'][i]]])
            visibles.append([1, 1, 1])

        self.img_names = img_names
        self.keypoints = kps
        self.visibles = visibles

        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_name)
        w, h = img.size
        kps = []
        w_ratio = self.img_size / w
        h_ratio = self.img_size / h
        for i in range(len(self.keypoints[idx])):
            if self.visibles[idx][i] == 0:
                # set keypoints to 0 when were not visible initially (so heatmap all 0s)
                kps.append([0.0, 0.0])
            else:
                ori_kp = self.keypoints[idx][i]
                x = ori_kp[0] * w_ratio * self.ratio
                y = ori_kp[1] * h_ratio * self.ratio
                kps.append([x, y])
        img = img.resize((self.img_size, self.img_size))

        heatmap = self.gen_heat_map(kps)
        return img, kps, heatmap

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    train_img_dir = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black"
    txt_file = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\kp_label.txt"
    im_resize, keypoints, heatmaps = ToothDataSet(train_img_dir, txt_file, 256)[0]
    print("keypoints: ", keypoints)
    print("im shape: ", im_resize.size)
    print("heatmaps shape: ", heatmaps[0].shape)

    import matplotlib.pyplot as plt
    im = im_resize.resize((64, 64))
    plt.imshow(im)
    # plt.imshow(heatmaps[0], alpha=0.5)
    # plt.imshow(heatmaps[1], alpha=0.5)
    plt.imshow(heatmaps[2], alpha=0.5)
    plt.show()
