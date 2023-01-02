import argparse as ap
import json
import pathlib
from logging import basicConfig

from vegdet.presenter import Presenter
from vegdet.inference import Inferrer
from vegdet.source import FileImageSource

parser = ap.ArgumentParser(
    "vegdet",
    "vegdet [file_glob]",
    "lightweight Vegetable detector based on efficientdet")

parser.add_argument("path")


def main():
    basicConfig(level=0)
    args = parser.parse_args()
    training = pathlib.Path(__file__).parents[1] / "training"
    with (training / "label_map.json").open("r") as fp:
        label_map = {int(i): j for i, j in json.load(fp).items()}

    infer = Inferrer(training / "model_veg_det_edl2.tflite", label_map)
    p = Presenter()
    src = FileImageSource(args.path)
    p.update_loop(src, infer)