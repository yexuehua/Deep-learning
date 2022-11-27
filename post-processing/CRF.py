import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


img = imread("im1.png")

# convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
anno_rgb = imread("anno1.png").astype(np.uint32)
anno_lb1 = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

# Convert the 32bit integer color to 1,2...labels.
colors, labels = np.unique(anno_lb1, return_inverse=True)

# But remove the all-0 black, that won't exist in MAP!
HAS_UNK = 0 in colors
if HAS_UNK:
    print(
        "Found a full-black pixel in annotation image, assuming it means 'unknow' label, and will not be present in the output!")
    print("If 0 is an acture label, consider writing your own code, or simply giving your labels only non-zero values.")
    colors = colors[1:]

colorize = np.empty((len(colors), 3), np.uint8)
colorize[:, 0] = (colors & 0x0000FF)
colorize[:, 1] = (colors & 0x00FF00) >> 8
colorize[:, 2] = (colors & 0xFF0000) >> 16

# compute the number of classes in the label image.
# we subtract one because the number shouldn't include the value which stands for "unknown" or "unsure".
n_labels = len(set(labels.flat)) - int(HAS_UNK)
print(n_labels, "labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
d.setUnaryEnergy(U)

# This adds the color-independent term, features are the location only
d.addPairwiseGaussian(sxy=(3, 3),
                      compat=3,
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

# This adds the color-dependent term, i.e. features are (x, y, r, g, b)
d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                       compat=10,
                       kernel=dcrf.DIAG_KERNEL,
                       normalization=dcrf.NORMALIZE_SYMMETRIC)

# do inference and compute MAP
Q = d.inference(5)
# find out the most probable class
MAP = np.argmax(Q, axis=0)

# convert the MAP back to the corresponding colors and save the image
# Note that there is no "unknown" here anymore, no matter what we had at first
MAP = colorize[MAP, :]
imsave("out.png", MAP.reshape(img.shape))

# Just randomly manually run the iterations
Q, tmp1, tmp2 = d.startInference()
for i in range(5):
    print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    d.stepInference(Q, tmp1, tmp2)