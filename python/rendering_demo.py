import chainer
import open3d as o3
import numpy as np
import csv
import cv2
from pprint import pprint 

import argparse
import sys
import os
import random

from rendering import Joint, Joints

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_op_lib = os.path.join(dir_path, 'openpose')
assert os.path.exists(abs_op_lib)
sys.path.insert(0, abs_op_lib)
try:
    from entity import params, JointType
except:
    print('Check the path for OpenPose Directory')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= '../data',help='relative data path from where you use this program')
    parser.add_argument('--static', default='static_data', help='static data location')
    args = parser.parse_args()

    # get directory of data (rgb, depth)
    data_path = os.path.join(dir_path, args.data)
    static_path = os.path.join(args.static)
    assert os.path.exists(data_path), "Could not find data directory in the path: {}".format(data_path)
    assert os.path.exists(static_path), "Could not find static data directory in the path: {}".format(static_path)
    print('Getting data from: {}'.format(data_path))
    
    # Translation matrix
    P_matrix_filename = os.path.join(static_path, 'T.csv')
    P = np.loadtxt(P_matrix_filename, delimiter=',')
    
    # Load room
    room_ply = os.path.join(static_path, 'room_mode_1.ply')
    pc_room = o3.read_point_cloud(room_ply)

    # pose path
    pose_path = os.path.join(data_path, 'pose')
    filename = random.choice(os.listdir(pose_path))
    while True:
        if filename.endswith('.csv'):
            break
        else:
            filename = random.choice(os.listdir(pose_path))
    
    tag = filename.split('.')[0]
    print('\nimage: ', tag)

    filename = "00164.csv"

    # get joints data and turn it into numpy array
    csv_path = os.path.join(pose_path, filename)
    raw_joints = np.loadtxt(csv_path, delimiter=',')
    
    joints = Joints(P, raw_joints)
    #pprint(joints.joints)
    pc_joints = joints.create_skeleton_geometry()
    mesh_frame = o3.create_mesh_coordinate_frame(size = 1000, origin = [0, 0, 0])
    pc_joints.append(pc_room)

    pc_joints.append(mesh_frame)

    o3.draw_geometries(pc_joints)