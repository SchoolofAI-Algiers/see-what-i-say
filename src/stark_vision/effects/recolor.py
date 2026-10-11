import numpy as np
import webcolors
import cv2

# class RecolorEffect:
#     def __init__(self):
#         pass

#     def apply(self,frame,mask,color):
#         changed_frame = np.copy(frame)

#         rgb_value = webcolors.name_to_rgb(color)
#         bgr_value = (rgb_value[2], rgb_value[1], rgb_value[0])

#         changed_frame[mask]=bgr_value
#         return changed_frame
        
    

class RecolorEffect:
    def __init__(self):
        pass

    def apply(self, frame, mask, color):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb_value = webcolors.name_to_rgb(color)
        bgr_value = (rgb_value[2], rgb_value[1], rgb_value[0])

        color_pixel = np.uint8([[bgr_value]])
        color_hsv = cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)

        new_hue = color_hsv[0, 0, 0]

        hsv[mask, 0] = new_hue

        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

    
recolor = RecolorEffect()
frame = cv2.imread("image.jpg")
mask = np.zeros((400,300),dtype=bool)
mask[100:200, 150:250]=True
result = recolor.apply(frame, mask, "red")

cv2.imshow("Recolor test", result)
cv2.waitKey(0)
cv2.destroyAllWindows()