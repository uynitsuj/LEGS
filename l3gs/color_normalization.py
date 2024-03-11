import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('home.jpg', cv2.IMREAD_GRAYSCALE)
# realsense_hists, zed_hists = [], []
# for i in range(181):
#     realsense_img = cv2.imread('../../Desktop/d455_zed_kitchen/realsense_compressed_image/image' + str(i).zfill(6) + '.png')
#     zed_img = cv2.imread('../../Desktop/d455_zed_kitchen/left_zed_left_image/image' + str(i).zfill(6) + '.png')
#     rs_hist = cv2.calcHist([realsense_img],[0],None,[256],[0,256])
#     zed_hist = cv2.calcHist([zed_img],[0],None,[256],[0,256])
#     realsense_hists.append(rs_hist), zed_hists.append(zed_hist)
# realsense_hists, zed_hists = np.array(realsense_hists), np.array(zed_hists)

# print(np.min(realsense_img), np.max(realsense_img), np.mean(realsense_img), np.median(realsense_img), np.std(realsense_img))
# print(np.min(zed_img), np.max(zed_img), np.mean(zed_img), np.median(zed_img), np.std(zed_img))
# print(realsense_img.shape, zed_img.shape)

# # cv2.imwrite('realsense_img.png', realsense_img)
# # cv2.imwrite('zed_img.png', zed_img)
# # cv2.imwrite('corrected_rs_image.png', cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(realsense_img, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR))

# print(realsense_hists.shape, zed_hists.shape)
# avg_rs_hist = np.mean(realsense_hists, axis=0)
# avg_zed_hist = np.mean(zed_hists, axis=0)

# print(avg_rs_hist.shape, avg_zed_hist.shape)
# print(np.min(avg_rs_hist), np.max(avg_rs_hist), np.mean(avg_rs_hist), np.median(avg_rs_hist), np.std(avg_rs_hist))
# print(np.min(avg_zed_hist), np.max(avg_zed_hist), np.mean(avg_zed_hist), np.median(avg_zed_hist), np.std(avg_zed_hist))

import PIL
from skimage import exposure

images_path = '../../Desktop/d455_zed_kitchen/'
which_images = 'left_zed_right_image'

realsense_img = cv2.imread(images_path + 'realsense_compressed_image/image' + str(0).zfill(6) + '.png')
for i in range(181):
    zed_img = cv2.imread(images_path + which_images + '/image' + str(i).zfill(6) + '.png')
    src, ref = zed_img, realsense_img
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, channel_axis=2)
    cv2.imwrite(images_path + f'corrected_{which_images}/image' + str(i).zfill(6) + '.png', matched)