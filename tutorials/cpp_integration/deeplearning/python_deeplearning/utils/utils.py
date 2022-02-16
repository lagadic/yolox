import numpy as np

from .bbox import BoundingBox


def postprocess_bboxes(img_net_size, original_img_size, bboxes, scores, classes, score_thresh):
    input_w, input_h = img_net_size
    output_w, output_h = original_img_size

    boxes = []
    bboxes = bboxes.astype(np.float32)
    for i in range(len(bboxes)):
        if scores[i] > score_thresh:
            x1, y1, x2, y2 = bboxes[i][:4]
            x1 *= output_w / input_w
            y1 *= output_h / input_h
            x2 *= output_w / input_w
            y2 *= output_h / input_h
            boxes.append(BoundingBox(x1, y1, x2, y2, scores[i], classes[i]))

    return boxes
