import os
import sys

import numpy as np
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import DataManagement

sys.path.insert(0, 'openpose')  # append path of openpose
from entity import params, JointType

def main():
    dm = DataManagement()
    after = dt(2018, 9, 9, 13, 7, 0)
    before = dt(2018, 9, 9, 13, 8, 0)
    datetimes = dm.get_datetimes_in(after, before)

    assert len(datetimes) == 1
    datetime = datetimes[0]

    print(datetime)

    item(dm, datetime)
    # pose(dm, datetime)
    


def pose(dm, datetime):

    data_path = dm.get_save_directory(datetime)
    data_path = os.path.join(data_path, 'pose')
    files = dm.natural_sort(os.listdir(data_path))  # sorted files

    joint_val = JointType.Nose.value

    print(JointType(joint_val))

    P_matrix_filename = os.path.join('static_data', 'T.csv')
    P = np.loadtxt(P_matrix_filename, delimiter=',')

    print("number of file: ", len(files))
    joint = np.zeros((len(files), 3))
    for i, f in enumerate(files):
        csv_path = os.path.join(data_path, f)
        points = np.loadtxt(csv_path, delimiter=',')
        point = points[joint_val]
        if (point == [0,0,0]).all():
            joint[i] = np.asarray([np.nan, np.nan, np.nan])
        else:
            joint[i] = convert2world(P, point)

    joint.shape
    roll_joint = np.rollaxis(joint, 1)
    roll_joint.shape

    show_location_map(roll_joint)



def item(dm, datetime):

    data_path = dm.get_save_directory(datetime)
    data_path = os.path.join(data_path, 'objects')

    print('objects: ', os.listdir(data_path))

    ob_name = 'chair'
    ob_path = os.path.join(data_path, ob_name)

    files = os.listdir(ob_path)
    print('number of files: ', len(files))
    filename = files[0]
    csv_path = os.path.join(ob_path, filename)

    P_matrix_filename = os.path.join('static_data', 'T.csv')
    P = np.loadtxt(P_matrix_filename, delimiter=',')

    ordered_files = dm.natural_sort(files)
    mean_of_points = np.zeros((len(ordered_files), 3))
    for i, fn in enumerate(ordered_files):
        csv_path = os.path.join(ob_path, fn)
        points = np.loadtxt(csv_path, delimiter=',')
        mean = points.mean(0)
        real_world = convert2world(P, mean)
        mean_of_points[i] = real_world

    mp = np.asarray(mean_of_points)
    mp = np.rollaxis(mp, 1)

    # show_movement(mp)

    show_location_map(mp)



def show_movement(points):

    plt.subplot(2, 2, 1)
    plt.plot(points[0], points[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.suptitle('Movement of object', fontsize=16)

    plt.subplot(2, 2, 2)
    plt.scatter(points[0], points[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def show_location_map(points):
    xedges = [i for i in range(-2000, 2000, 50)]
    yedges = [i for i in range(-2000, 2000, 50)]

    x = points[0][~np.isnan(points[0])]
    y = points[1][~np.isnan(points[1])]
    # z = points[2][~np.isnan(points[2])]

    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T


    fig = plt.figure()

    ax = fig.add_subplot(111, title='object heat map',
                        aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    # non uniform image:

    # ax = fig.add_subplot(111, title='NonUniformImage: interpolated',
    #                      aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    # im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # im.set_data(xcenters, ycenters, H)
    # ax.images.append(im)
    plt.show()


def convert2world(P, coord):
    '''
    Convert from Camera coordniate to World coordinate (according to P)
    '''
    _coord = np.concatenate([np.asarray(coord), [1.000]])
    _P = np.array(P)
    #FIXME: Remove this when P is fixed
    rotate = np.array([[1,0,0,0], [0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    n = rotate.dot(_coord)
    return _P.dot(n)[:3]



if __name__ == '__main__':
    main()