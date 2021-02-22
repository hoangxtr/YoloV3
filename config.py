import numpy as np

STRIDES = np.array([8, 16, 32])

YOLO_ANCHORS = [[[10,  13], [16,   30], [33,   23]],
                [[30,  61], [62,   45], [59,  119]],
                [[116, 90], [156, 198], [373, 326]]]
ANCHORS = (np.array(YOLO_ANCHORS)/416)