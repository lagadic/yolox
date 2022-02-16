import os
import numpy as np

from tensorflow.keras.models import load_model
from .utils.utils import postprocess_bboxes


class Inference:
    def __init__(self, model_path):
        assert os.path.isdir(model_path), \
            "Model {} does not exist.".format(model_path)

        # Init tf model
        self.model = load_model(model_path)

    def predict(self, image, origin_h, origin_w, net_size, score_thresh):

        images = np.expand_dims(image, axis=0)
        bboxes, scores, classes, valid_detections = self.model.predict(images)

        bboxes = bboxes[0][:valid_detections[0]]
        scores = scores[0][:valid_detections[0]]
        classes = classes[0][:valid_detections[0]]

        return postprocess_bboxes((net_size, net_size), (origin_w, origin_h), bboxes, scores, classes, score_thresh)
