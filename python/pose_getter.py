import cv2
import chainer
import open3d as o3
import numpy as np

import argparse
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_op_lib = os.path.join(dir_path, 'openpose')
assert os.path.exists(abs_op_lib)
sys.path.insert(0, abs_op_lib)
try:
    from entity import params, JointType
    from pose_detector import PoseDetector, draw_person_pose
except:
    print('Check the path for OpenPose Directory')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= 'data',help='relative data path from where you use this program')
    parser.add_argument('--save', default= 'pose',help='relative saving directory from where you use this program')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # set up for Chainer
    chainer.config.enable_backprop = False
    chainer.config.train = False

    # get directory of data (rgb, depth)
    data_path = os.path.join(dir_path, args.data)
    save_path = os.path.join(data_path, args.save)
    assert os.path.exists(data_path), "Could not find data directory in the path: {}".format(data_path)
    print('Getting data from: {}'.format(data_path))
    if not os.path.exists(save_path):
        print('Making a save directory in: {}'.format(save_path))
        os.makedirs(save_path)

    rgb_path = os.path.join(data_path, 'rgb')
    depth_path = os.path.join(data_path, 'depth')

    # load model
    pose_detector = PoseDetector("posenet", 
                                 os.path.join(abs_op_lib, 
                                 "models/coco_posenet.npz"), 
                                 device=args.gpu)

    # Loop:
    for filename in os.listdir(rgb_path):
        if filename.endswith(".png"): 
            tag = filename.split('.')[0]
            print('\nimage: ', tag)

            # find the corresponding depth image
            rgb_img = os.path.join(rgb_path, filename)
            depth_img = os.path.join(depth_path, filename)
            if not os.path.exists(depth_img):
                print('Could not find corresponding depth image in: {}'.format(depth_img))
                continue
        
            # read image
            img = cv2.imread(rgb_img)

            # inference
            poses, scores = pose_detector(img)
            # print(poses)
            if len(poses) > 2:
                print('too many poses!')
                continue

            # use open3d api for data pipeline
            rgb_raw = o3.read_image(rgb_img)
            depth_raw = o3.read_image(depth_img)
            rgbd_image = o3.create_rgbd_image_from_color_and_depth(rgb_raw, depth_raw)
            
            # open3d.Image to numpy array
            depths = np.array(rgbd_image.depth)

            for i, pose in enumerate(poses):
                csv_name = tag + '_' + str(i) + '.csv' if i > 0 else tag + '.csv'

                joints = np.zeros((len(JointType), 3))

                for i, joint in enumerate(pose):
                    x, y = int(joint[0]), int(joint[1])
                    depth = depths[y][x]
                    #print('x: {}, y: {}, depth: {}'.format(x, y, depth))
                    if depth != 0.0:
                        joints[i] = np.array([x, y, depth])
                
                # print(joints)
                

                csv_path = os.path.join(save_path, csv_name)
                np.savetxt(csv_path, joints, delimiter=",")
                    

            continue
    

    