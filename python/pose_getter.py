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


class Open3D_Chain:
    '''
    Open3D Chain for easy rendering
    '''
    def __init__(self):
        self.camera_intrinsic = o3.read_pinhole_camera_intrinsic("static_data/d415.json")
        self.K = np.asarray(self.camera_intrinsic.intrinsic_matrix)

    def change_image(self, rgb_path, depths_path):
        assert os.path.exists(rgb_path), 'Could not find corresponding rgb image in: {}'.format(rgb_path)
        assert os.path.exists(depths_path), 'Could not find corresponding depth image in: {}'.format(depths_path)
        self.rgb =  self.read_image(rgb_path)
        self.depths =  self.read_image(depths_path)

    def read_image(self, path):
        return np.asarray(o3.read_image(path))

    def calc_xy(self, x, y, z):
        '''
        K: intrinsic matrix
        x: pixel value x
        y: pixel value y
        z: mm value of z
        '''
        fx = self.K[0][0]
        fy = self.K[1][1]
        u0 = self.K[0][2]
        v0 = self.K[1][2]

        _x = (x - u0) * z / fx
        _y = (y - v0) * z / fy
        return _x, _y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= '../data',help='relative data path from where you use this program')
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

    # camera params
    o3_chain = Open3D_Chain()

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
            o3_chain.change_image(rgb_img, depth_img)

            # inference
            poses, scores = pose_detector(o3_chain.rgb)
            # print(poses)
            if len(poses) > 2:
                print('too many poses!')
                continue

            for i, pose in enumerate(poses):
                csv_name = tag + '_' + str(i) + '.csv' if i > 0 else tag + '.csv'

                joints = np.zeros((len(JointType), 3))

                for i, joint in enumerate(pose):
                    x, y = int(joint[0]), int(joint[1])
                    Z = o3_chain.depths[y][x]

                    if Z != 0.0:
                        # Depth = 0 means that depth data was unavailable
                        X, Y = o3_chain.calc_xy(x, y, Z)
                        joints[i] = np.asarray([X, Y, Z])
                        print('x: {}, y: {}, depth: {}'.format(X, Y, Z))
                
                csv_path = os.path.join(save_path, csv_name)
                np.savetxt(csv_path, joints, delimiter=",")
    