import sys
import os

import chainer
import numpy as np
import open3d as o3
import cv2
from pyrs import PyRS
import matplotlib.pyplot as plot
from chainercv import utils

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_maskrcnn = os.path.join(dir_path, 'maskrcnn')
assert os.path.exists(abs_maskrcnn)
sys.path.insert(0, abs_maskrcnn)
try:
    from mask_rcnn_train_chain import MaskRCNNTrainChain
    from utils.bn_utils import freeze_bn, bn_to_affine
    from mask_rcnn_resnet import MaskRCNNResNet
    from utils.vis_bbox import vis_bbox
except:
    print('Check the path for Mask-RCNN Directory')


if __name__ == '__main__':

    test_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, \
    57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_label_names = ('background',  # class zero
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'mirror', 'dining table', 'window', 'desk','toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    )

    roi_size = 14
    modelfile = os.path.join(abs_maskrcnn, 'modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz')
    roi_align = True

    model = MaskRCNNResNet(n_fg_class = 80,
                        roi_size = roi_size,
                        pretrained_model = modelfile,
                        n_layers = 50,  # resnet 50 layers (not 101 layers)
                        roi_align = roi_align,
                        class_ids = test_class_ids)
    chainer.serializers.load_npz(modelfile, model)

    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()
    bn_to_affine(model)


    
    name = 'cup'
    w = 1280
    h = 720


    chair_index = coco_label_names.index(name)
    with PyRS(w=w, h=h) as pyrs:
        print('Modes:')
        print('\tExit:\tq')

        preset = pyrs.get_depths_preset()
        preset_name = pyrs.get_depths_preset_name(preset)
        print('Preset: ', pyrs.get_depths_preset_name(preset))

        while True:
            # Wait for a coherent pair of frames: depth and color
            pyrs.update_frames()

            # Get images as numpy arrays
            color_image = pyrs.get_color_image()
            depths_image = pyrs.get_depths_frame()

            color = color_image.swapaxes(2, 1).swapaxes(1, 0)
            bboxes, labels, scores, masks = model.predict([color])
            bbox, label, score, mask = bboxes[0], np.asarray(labels[0], dtype=np.int32), scores[0], masks[0]

            # use chair as example:
            label_index = np.where(label == chair_index)
            if label_index[0].any():
                label_index = label_index[0][0]
                
                chair_mask = mask[label_index]
                chair_depth = np.multiply(depths_image, chair_mask)

                y1, x1, y2, x2 = [int(n) for n in bbox[label_index]]
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(color_image, name, (x1 + 10, y1 + 10), 0, 0.3, (0,255,0))

            else:
                chair_depth = np.zeros([h, w])
            
            chair_image = cv2.applyColorMap(cv2.convertScaleAbs(chair_depth, None, 0.5, 0), cv2.COLORMAP_JET)



            images = np.hstack((color_image, chair_image))
            
            # Show image
            cv2.namedWindow('Chair Tracker', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Chair Tracker', images)
            key = cv2.waitKey(10)

            if key == ord('q'):
                # end OpenCV loop
                break
            elif key == ord('p'):
                # save rgb and depths
                cv2.imwrite("color_chair.png", color_image)
                cv2.imwrite("depth_chair.png", depths_image)
                cv2.imwrite("tracker.png", images)