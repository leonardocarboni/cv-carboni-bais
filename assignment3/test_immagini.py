import cv2 as cv
import numpy as np


data = np.load('data/immagini.npz')['immagini']

for i, image in enumerate(data):
    cv.imshow(f'frame {i}', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

