import argparse
import sys
import os

import numpy as np
import chainer


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_op_lib = os.path.join(dir_path, 'openpose')
assert os.path.exists(abs_op_lib)
sys.path.insert(0, abs_op_lib)
try:
    from open3d_chain import Open3D_Chain
    from entity import params, JointType
    from pose_detector import PoseDetector, draw_person_pose
except:
    print('Check the path for OpenPose Directory')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= '/mnt/extHDD/raw_data/20180722_1643/',help='relative data path from where you use this program')
    parser.add_argument('--save', default= '/mnt/extHDD/save_data/20180722_1643/pose',help='relative saving directory from where you use this program')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # set up for Chainer
    chainer.config.enable_backprop = False
    chainer.config.train = False

    # get directory of data (rgb, depth)
    data_path = os.path.join("/", args.data)
    save_path = os.path.join("/", args.save)
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

    # sort rgb files before looping
    # order matters!
    files = os.listdir(rgb_path)
    #files = [int(''.join(filter(str.isdigit, f))) for f in files]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Loop:
    pose_num = 0
    for filename in files:
        if filename.endswith(".png"): 
            print('\nimage: ', filename)
            # find the corresponding depth image
            rgb_img = os.path.join(rgb_path, filename)
            depth_img = os.path.join(depth_path, filename)
            if not os.path.exists(depth_img):
                print('Could not find corresponding depth image in: {}'.format(depth_img))
                continue

            # read image
            o3_chain.change_image(rgb_img, depth_img)

            # inference
            poses, scores = pose_detector(o3_chain.get_rgb())

            for i, pose in enumerate(poses):
                csv_name = str(pose_num) + '.csv'

                joints = np.zeros((len(JointType), 3))

                for i, joint in enumerate(pose):
                    x, y = int(joint[0]), int(joint[1])
                    Z = o3_chain.get_depths()[y][x]

                    if Z != 0.0:
                        # Depth = 0 means that depth data was unavailable
                        X, Y = o3_chain.calc_xy(x, y, Z)
                        joints[i] = np.asarray([X, Y, Z])
                        # print('x: {}, y: {}, depth: {}'.format(X, Y, Z))
                
                csv_path = os.path.join(save_path, csv_name)
                np.savetxt(csv_path, joints, delimiter=",")
                
                pose_num += 1
    