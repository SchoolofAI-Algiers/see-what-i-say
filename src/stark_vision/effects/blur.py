import cv2
import numpy as np

class BlurEffect:
    def __init__(self):
        pass

    def apply(self,frame,mask):
        blurred = np.copy(frame)
        all_blurred = cv2.GaussianBlur(frame,(31,31),0)
        blurred[mask]=all_blurred[mask]
        return blurred

blurObject = BlurEffect()

frame = cv2.imread("image.jpg")
mask = np.zeros((400,300),dtype=bool)
mask[100:200,150:250] = True
blurred_img = blurObject.apply(frame,mask)

cv2.imshow("Blur test", blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()