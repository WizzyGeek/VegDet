
import logging

try:
    import tflite_runtime.interpreter as tflite
except (ImportError, ModuleNotFoundError): # Incase of a non-pip install
    try:
        import tensorflow.lite as tflite # type: ignore # noqa
    except:
        print("Tensorflow lite runtime not installed!")
        raise

import numpy as np

inf_logger = logging.getLogger("vegdet.Inferrer")

class Inferrer:
    def __init__(self, model_path, label_map):
        self.interpreter = inter = tflite.Interpreter(model_path=str(model_path), num_threads=8) # Most cpu have 4+hyper-V, saturate the CPU
        inter.allocate_tensors() # expect inference soon
        assert len(inter.get_signature_list()) == 1 # Loose check for efficientdet-lite2 for now, we dont want this class to be scalable
        self.runner = inter.get_signature_runner()
        self.label_map = label_map
        self.label_map[0] = "???"
        _, self.height, self.width, _ = inter.get_input_details()[0]['shape']

    def run(self, pre_proc_img, thresh=0.2):
        image_tensor = pre_proc_img[np.newaxis, :] # broadcast array from 3d with 3 channels to 4d with 1 image
        out = self.runner(images=image_tensor) # assuming this is one of the valid models, images, is the input name
        inf_logger.debug("%s", out) # comment this out later

        class_ids = np.squeeze(out["output_2"]) # output 0 is just count, not needed
        boxes = np.squeeze(out["output_3"])
        lbl = self.label_map # for speed, make closure to attribute for map lambda
        return list(
            map(
                lambda s: (s[1], lbl[int(class_ids[s[0]])], boxes[s[0]]),
                filter(
                    lambda s: s[1] >= thresh,
                    enumerate(np.squeeze(out["output_1"])) # Enumerate score tensor
                )
            )
        )

