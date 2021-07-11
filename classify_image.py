from keras.models import load_model
import cv2

model_path = 'classify_image_model.hd5' #TODO: add to config
image_path = 'circle_imgs/img2021-07-08_11:12:24.653021_crop.jpg' #TODO: probably a command line arg?
# or this should just be a function

if __name__ == '__main__':
    model = load_model(model_path)
    img = cv2.imread(image_path)
