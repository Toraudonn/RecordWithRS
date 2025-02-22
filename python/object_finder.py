import argparse
import sys
import os

import chainer
from chainercv import utils
import numpy as np
import matplotlib.pyplot as plot

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_maskrcnn = os.path.join(dir_path, 'maskrcnn')
from maskrcnn import MaskRCNNTrainChain
from maskrcnn import freeze_bn, bn_to_affine
from maskrcnn import MaskRCNNResNet
from maskrcnn import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--modelfile', default= os.path.join(abs_maskrcnn, 'modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz'))
    parser.add_argument('--image', type=str)
    parser.add_argument('--roi_size', '-r', type=int, default=14, help='ROI size for mask head input')
    parser.add_argument('--roialign', action='store_false', default=True, help='default: True')
    parser.add_argument('--contour', action='store_true', default=True, help='visualize contour')
    parser.add_argument('--background', action='store_true', default=False, help='background(no-display mode)')
    parser.add_argument('--bn2affine', action='store_true', default=True, help='batchnorm to affine')
    parser.add_argument('--extractor', choices=('resnet50','resnet101'),
                        default='resnet50', help='extractor network')
    args = parser.parse_args()

    #network class id --> coco label id
    test_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, \
    57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    if args.background:
        import matplotlib
        matplotlib.use('Agg')
    if args.extractor=='resnet50':
        model = MaskRCNNResNet(n_fg_class=80, roi_size=args.roi_size, pretrained_model=args.modelfile, n_layers=50, roi_align=args.roialign, class_ids=test_class_ids)
    elif args.extractor=='resnet101':
        model = MaskRCNNResNet(n_fg_class=80, roi_size=args.roi_size, pretrained_model=args.modelfile, n_layers=101, roi_align=args.roialign, class_ids=test_class_ids)

    chainer.serializers.load_npz(args.modelfile, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    if args.bn2affine:
        bn_to_affine(model)
    img = utils.read_image(args.image, color=True)
    print(img.shape)
    bboxes, labels, scores, masks = model.predict([img])

    bbox, label, score, mask = bboxes[0], np.asarray(labels[0],dtype=np.int32), scores[0], masks[0]

    coco_label_names=('background',  # class zero
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'mirror', 'dining table', 'window', 'desk','toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    )
    vis_bbox(
        img, bbox, label=label, score=score, mask=mask, label_names=coco_label_names, contour=args.contour, labeldisplay=True)
    plot.show()
    # filename = "output.png"
    # plot.savefig(filename)


if __name__ == '__main__':
    main()
