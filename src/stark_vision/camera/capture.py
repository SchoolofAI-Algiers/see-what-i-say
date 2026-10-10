
import cv2 
import time 

class CameraCapture:

    def __init__(self):
        self.cap = cv2.VideoCapture(0) 
        self.nb_frames = 0
        self.start_time = 0
        self.end_time = 0

    def capture(self):
        print ("Camera is recording ...")
        value = True
        self.start_time = time.perf_counter()
        while value == True:
            value, image=self.cap.read()
            if value == True: self.nb_frames = self.nb_frames + 1
            cv2.imshow('frame',image)
            if cv2.waitKey(1) == ord("p"):
                self.stop()
                self.end_time = time.perf_counter()
                value = False
        self.calculations()

    def stop(self):
        print("Capture Stopped!")
        self.cap.release()
        cv2.destroyAllWindows()

    def calculations(self) :
        elapsed_time = self.end_time - self.start_time
        print ("elapsed time = ",self.end_time - self.start_time)
        print ("nb_frames",self.nb_frames)
        avg_fps = self.nb_frames / elapsed_time
        print ("avg FPS = ", avg_fps )

if __name__ == "__main__":
    camera = CameraCapture()
    camera.capture()

