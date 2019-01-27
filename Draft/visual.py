import cv2
from skimage.feature import hog


def HOG_features(img, n_hogs_per_dim = 4):
    pixels_per_cell_h = img.shape[0] // n_hogs_per_dim
    pixels_per_cell_w = img.shape[1] // n_hogs_per_dim
    fd = hog(img, orientations=8,
              pixels_per_cell=(pixels_per_cell_h, pixels_per_cell_w),
              cells_per_block=(1, 1), visualize=False, multichannel=True)
    f_names = ['HOG' + str(i).zfill(2) for i in range(len(fd))]
    return fd, f_names


