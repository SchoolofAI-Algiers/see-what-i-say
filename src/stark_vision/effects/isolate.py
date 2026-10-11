import numpy as np
import cv2

class IsolateEffect:
    def __init__(self):
        pass

    def apply(self,frame,mask):
        copied = np.copy(frame)
        new_frame = np.zeros_like(frame)
        new_frame[mask] = copied[mask]
        return new_frame
        
isolateObject = IsolateEffect()
frame = cv2.imread("image.jpg")
mask = np.zeros((400,300),dtype=bool)
mask[100:200,150:250] = True
isolated_img = isolateObject.apply(frame,mask)

cv2.imshow("Isolate test", isolated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()