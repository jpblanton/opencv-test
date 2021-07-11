from pathlib import Path

from decouple import config
import numpy as np
import cv2


def decide_color(img, g_l, g_u, r_l, r_u):

    grn_mask = cv2.inRange(img, grn_lower, grn_upper)
    grn_out = cv2.bitwise_and(img, img, mask=grn_mask)
    red_mask = cv2.inRange(img, red_lower, red_upper)
    red_out = cv2.bitwise_and(img, img, mask=red_mask)

    grn_sum = grn_out.sum()
    red_sum = red_out.sum()
    # with light on, img_sum > 100 million
    # with off, more like 10 million or less
    # probably won't work in real env
    # but ok for now - can then use greater of red/green to predict color
    img_sum = img.sum()

    if img_sum > 50_000_000:
        # too bright
        # TODO: solve this - actually, decide if necessary when inside tent
        return (None, grn_sum, red_sum, img_sum)
    else:
        if grn_sum > red_sum:
            return ('grn', grn_sum, red_sum, img_sum)
        else:
            return ('red', grn_sum, red_sum, img_sum)


# bounds are probably way different with the light and all
# need to open up picture and examine for threshold
# or use keras for these now 45x45 images to train
grn_lower = np.array(config('GRN_LOWER').split(','), dtype='uint8')
grn_upper = np.array(config('GRN_UPPER').split(','), dtype='uint8')
red_lower = np.array(config('RED_LOWER').split(','), dtype='uint8')
red_upper = np.array(config('RED_UPPER').split(','), dtype='uint8')

RADIUS = 21
HEIGHT = 480
WIDTH = 720 # constant for all no need to recalculate

image_dir = Path('/mnt/c/Users/James/test_camera_pics/set2/')
output_dir = 'circle_imgs'
tmp_lst = list(image_dir.glob('*.jpg'))

for i, pth in enumerate(tmp_lst):
    img = cv2.imread(str(pth))
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (RADIUS, RADIUS), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    circle_img = cv2.circle(mask, maxLoc, RADIUS, (255, 0, 0), 2)
    masked_data = cv2.bitwise_and(img, img, mask=circle_img)
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0][0])
    crop = orig[y:y+h,x:x+w]
    cv2.imwrite(str(Path(output_dir) / pth.name), img)
    cv2.imwrite(str(Path(output_dir) / pth.stem) + '_crop.jpg', crop)
    color, _, _, _ = decide_color(crop, grn_lower, grn_upper, red_lower, red_upper)
