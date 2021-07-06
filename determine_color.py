from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

def encode_bounds(g_l, g_u, r_l, r_u):
    g = ','.join([':'.join([str(j) for j in i]) for i in zip(g_l, g_u)])
    r = ','.join([':'.join([str(j) for j in i]) for i in zip(r_l, r_u)])
    return g + ';' + r

def decode_bounds(s):
    # TODO: split by ; then by , then by :
    pass

# current thought is to use https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
# with inRange to get the mask for green/red light (depending)
# will display to check, but I think we can use the masked image to decide which is better?
# yes, this is good. can sum() mask. over 100k is result for correct images
# for incorrect, red image green mask had 0
# green image red mask had 80kish - might want to adjust
# don't forget to use the well lit ones

img_path = r"/mnt/c/Users/James/test_camera_pics/"
img_path_red = r"/mnt/c/Users/James/test_camera_pics/img2021-07-04 03-23-12.439031.jpg"
img_path_grn = r"/mnt/c/Users/James/test_camera_pics/img2021-07-04 03-24-12.979465.jpg"

grn_lower = np.array([0, 86, 0], dtype='uint8')
grn_upper = np.array([220, 255, 50], dtype='uint8')
red_lower = np.array([0, 0, 86], dtype='uint8')
red_upper = np.array([150, 220, 255], dtype='uint8')

# should save at least the file name/true/light_on in a base file
results_df = pd.read_csv('base_results_table.csv')
# to save bounds, could write quick method
# to str().zfil(3) all and write that into filename
# hard to write out file name then if need be

for idx, row in results_df.iterrows():

    img = cv2.imread(row['full_path'])

    grn_mask = cv2.inRange(img, grn_lower, grn_upper)
    grn_out = cv2.bitwise_and(img, img, mask = grn_mask)
    red_mask = cv2.inRange(img, red_lower, red_upper)
    red_out = cv2.bitwise_and(img, img, mask = red_mask)

    grn_sum = grn_out.sum()
    red_sum = red_out.sum()
    # with light on, img_sum > 100 million
    # with off, more like 10 million or less
    # probably won't work in real env
    # but ok for now - can then use greater of red/green to predict color
    img_sum = img.sum()

    results_df.loc[idx, 'grn_sum'] = grn_sum
    results_df.loc[idx, 'red_sum'] = red_sum
    results_df.loc[idx, 'img_sum'] = img_sum

    # cv2.imwrite(out_name.format(fn=pth.stem, color='green'), grn_out)
    # cv2.imwrite(out_name.format(fn=pth.stem, color='red'), red_out)

bounds = encode_bounds(grn_lower, grn_upper, red_lower, red_upper)
results_df.columns.name = bounds

results_df.to_csv(f'{str(int(time.time()))}_results.csv')
