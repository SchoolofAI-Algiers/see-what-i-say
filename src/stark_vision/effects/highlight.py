import numpy as np
import cv2

class HighlightEffect:
    def apply(self, frame, mask):
        result = np.copy(frame)

        darkened = (frame * 0.35).astype(np.uint8)
        result[~mask] = darkened[~mask]
        result[mask] = frame[mask]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            result,
            contours,
            -1,
            (0, 255, 255),
            3
        )

        return result
        

highlightObject = HighlightEffect()
frame = cv2.imread("image.jpg")
mask = np.zeros((400,300),dtype=bool)
mask[100:200,150:250] = True
highlight_img = highlightObject.apply(frame,mask)

cv2.imshow("Highlight test", highlight_img)
cv2.waitKey(0)
cv2.destroyAllWindows()