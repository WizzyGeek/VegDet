from __future__ import annotations

from typing import TYPE_CHECKING
from time import time

import cv2

from vegdet.preprocess import preprocess

if TYPE_CHECKING:
    from vegdet.source import ImageSource
    from vegdet.classifier import Classifier

class ClassifierPresenter:
    WINDOW_NAME = "Classifier Results"

    def __init__(self) -> None:
        pass

    def show_ret(self, orig_img, labels_score_normalised):
        top_3 = list(map(
            lambda s: "{0:.2%} - ".format(s[0]) + s[1],
            labels_score_normalised[-3:][::-1]
        ))
        print(*top_3, sep="\n")
        cv2.putText(orig_img, top_3[0], (0, orig_img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(self.WINDOW_NAME, orig_img)

    def update_loop(self, src: ImageSource, classifier: Classifier, thresh=0.0):
        img = src.get_next()
        delay = src.delay
        keys = {27, ord("q")}
        try:
            while img is not None:
                retw = classifier.run(preprocess(img, *classifier.dim))
                ret = list(filter(lambda s: s[0] >= thresh, map(lambda s: (s[0] / 255, s[1]), retw)))
                # print(retw, ret)
                self.show_ret(img, ret)
                if cv2.getWindowProperty(self.WINDOW_NAME, 0) < 0: # WINDOW FULL SCREEN property, but it goes to -1 when window closed
                    break
                if cv2.waitKey(delay) in keys:
                    break
                img = src.get_next()
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()