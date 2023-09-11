import os.path

import torch

from coco_dataset import CocoDataset

root_data = "./data/coco"
path2trainList = os.path.join(root_data, "trainvalno5k.txt")
coco_train = CocoDataset(path2trainList)
print(len(coco_train))
img, labels, path2img = coco_train[1]
print("image size:", img.size, type(img))
print("label shape:", labels.shape, type(labels))
print("label \n", labels)

path2valList = os.path.join(root_data, "5k.txt")
coco_val = CocoDataset(path2valList)
print(len(coco_val))
img_val, labels_val, path2img_val = coco_val[7]
print("image size:", img_val.size, type(img_val))
print("label shape:", labels_val.shape, type(labels_val))
print("label \n", labels_val)

import matplotlib.pylab as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
import random

path2cocoNames = "./data/coco.names"
fp = open(path2cocoNames, "r")
coco_names = fp.read().split("\n")[:-1]
print("number of classes:", len(coco_names))
print(coco_names)

def rescale_bbox(bb, W, H):
    x, y, w, h = bb
    return [x*W, y*H, w*W, h*H]

# get random color index
COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")
fnt = ImageFont.truetype("./data/coco/FreeMono.ttf", 16)
def show_img_bbox(img, targets):
    if torch.is_tensor(img):
        img = to_pil_image(img)
    if torch.is_tensor(targets):
        targets = targets.numpy()[:,1:]
    W, H = img.size
    draw = ImageDraw.Draw(img)

    for tg in targets:
        id_ = int(tg[0])
        bbox = tg[1:]
        # bbox has been rescaled to 0-1, so it should be scaled back
        bbox = rescale_bbox(bbox, W, H)
        xc, yc, w, h = bbox
        color = [int(c) for c in COLORS[id_]]
        name = coco_names[id_]
        draw.rectangle(((xc-w/2, yc-h/2), (xc+w/2, yc+h/2)), outline=tuple(color), width=3)
        draw.text((xc-w/2, yc-h/2), name, font=fnt, fill=(255,255,255,0))

    plt.imshow(np.array(img))
    plt.show()

np.random.seed(2)
rnd_ind = np.random.randint(len(coco_train))
img, labels, path2img = coco_train[rnd_ind]
print(img.size, labels.shape)
plt.rcParams['figure.figsize'] = (20,10)
show_img_bbox(img, labels)