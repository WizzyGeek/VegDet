from tkinter.tix import WINDOW
import cv2

from vegdet.source import ImageSource
from vegdet.inference import Inferrer
from vegdet.preprocess import preprocess

class Presenter:
    WINDOW_NAME = "Detector Results"

    def __init__(self) -> None:
        pass

    def show_ret(self, orig_img, results):
        x, y = orig_img.shape[:2]
        for score, label, box in results:
            ymin, xmin, ymax, xmax = box
            cv2.rectangle(orig_img, (int(xmin * x), int(ymin * y)), (int(xmax * x), int(ymax * y)), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(orig_img, label + "{0:%}".format(score), (int(xmin * x)+15, int(ymax * y)-20), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(self.WINDOW_NAME, orig_img)

    def update_loop(self, src: ImageSource, infer: Inferrer, thresh=0.2):
        img = src.get_next()
        delay = src.delay
        keys = {ord("q"), 27}
        try:
            while img is not None:
                ret = infer.run(preprocess(img, infer.width, infer.height), thresh)
                print(ret, infer.label_map)
                self.show_ret(img, ret)
                if cv2.getWindowProperty(self.WINDOW_NAME, 0) < 0: # WINDOW FULL SCREEN property, but it goes to -1 when window closed
                    break
                if cv2.waitKey(delay) in keys:
                    break
                img = src.get_next()
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()