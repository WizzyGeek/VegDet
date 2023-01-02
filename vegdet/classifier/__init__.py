import argparse as ap
import imp
from pathlib import Path
from pprint import pprint

import numpy as np
import cv2

from vegdet.inference import tflite
from vegdet.preprocess import preprocess

labels = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

class Classifier:
    def __init__(self, model_path):
        self.interpreter = inter = tflite.Interpreter(model_path=str(model_path))
        self.input_tensor_index = inter.get_input_details()[0]["index"]
        self.dim = tuple(inter.get_input_details()[0]["shape"][1:3][::-1])
        self.output_tensor_index = inter.get_output_details()[0]["index"]
        inter.allocate_tensors()
        out_fn = inter.tensor(self.output_tensor_index)
        self.get_scores = lambda: [(score, labels[idx]) for idx, score in enumerate(out_fn()[0])]

    def run(self, image):
        self.interpreter.set_tensor(self.input_tensor_index, image[np.newaxis, :])
        self.interpreter.invoke()
        scores = self.get_scores()
        scores.sort(key=lambda n: n[0])
        return scores

parser = ap.ArgumentParser(
    "vegclass",
    "vegclass [path]",
    "lightweight Vegetable classifier based on efficientnet")

parser.add_argument("path")

def classify():
    args = parser.parse_args()
    img = cv2.imread(args.path)
    classifier = Classifier(Path(__file__).parents[2] / "training" / "model.tflite")
    processed = preprocess(img, *classifier.dim)
    scores = classifier.run(processed)
    pprint(scores[::-1])
    cv2.putText(img, scores[-1][1] + ": {0:%}".format(scores[-1][0]/255), (0, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('newwin', img)
    cv2.waitKey(10000)

if __name__ == "__main__":
    classify()