import numpy
import cv2
from scipy import misc
import matplotlib.pyplot as plt

jpg_file="C://Users//song//Downloads//A1B1C1D1E1_51THY1820180723105819661.jpg"
out_jpg_file="C://Users//song//Downloads//A1B1C1D1E1_51THY1820180723105819661_mask.jpg"
# image = cv2.imread(jpg_file)

image = misc.imread(jpg_file, flatten=True)
#
binary_map = image * 255
cv2.imwrite(out_jpg_file, binary_map)
