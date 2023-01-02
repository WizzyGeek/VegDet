import cv2

from vegdet.source import ImageSource
from vegdet.inference import Inferrer
from vegdet.preprocess import preprocess

class Presenter:
    def __init__(self) -> None:
        pass

    def show_ret(self, orig_img, results):
        x, y = orig_img.shape[:2]
        for score, label, box in results:
            ymin, xmin, ymax, xmax = box
            cv2.rectangle(orig_img, (int(xmin * x), int(ymin * y)), (int(xmax * x), int(ymax * y)), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(orig_img, label + "{0:%}".format(score), (int(xmin * x)+15, int(ymax * y)-20), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", orig_img)

    def update_loop(self, src: ImageSource, infer: Inferrer, thresh=0.2):
        img = src.get_next()
        delay = src.delay
        key = ord("q")
        while img is not None:
            ret = infer.run(preprocess(img, infer.width, infer.height), thresh)
            print(ret, infer.label_map)
            self.show_ret(img, ret)
            if cv2.waitKey(delay) == key:
                break
            img = src.get_next()
        # cv2.destroyAllWindows()