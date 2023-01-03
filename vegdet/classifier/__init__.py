import argparse as ap
from pathlib import Path

import numpy as np

from vegdet.inference import tflite
from vegdet.source import FileImageSource
from vegdet.classifier.presenter import ClassifierPresenter

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
    src = FileImageSource(args.path)
    classifier = Classifier(Path(__file__).parents[2] / "training" / "model.tflite")
    presenter = ClassifierPresenter()
    presenter.update_loop(src, classifier)

if __name__ == "__main__":
    classify()