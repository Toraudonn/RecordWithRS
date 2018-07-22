import os
import sys
import time
from datetime import datetime as dt

''' 
# Objective:

Better management for my project\'s data format

# Directory format:
    root_dir:
        - DATE_TIME:
            - rgb:
                - files
            - depth:
                -files
## Details:

Directory name:

- DATE_TIME: "%Y%m%d_%H%M"
    - separated by minutes


# TODO:

- Getting data and saving data differs...
- Data path is always not in this format

'''

raw_data = '/mnt/extHDD/raw_data/'
save_data = 'mnt/extHDD/save_data/'
rgb = 'rgb'
depth = 'depth'


class FileManagement:
    def __init__(self, root=raw_data):
        self.root = root
        #TODO: Get only directories you are working on:
        # i.e., only get directories that are between two date_time
        self.datetime_dirs = sorted(os.listdir(self.root))  # algorithm cost?
        self.datetimes = [self._string2datetime(n) for n in self.datetime_dirs]

    def get_datetime_dirs(self):
        '''returns a list of strings...'''
        return self.datetime_dirs
    
    def get_datetimes(self):
        '''returns a list of datetime'''
        return self.datetimes

    def check_path_exists(self, path):
        if os.path.exists(path):
            return True
        else:
            print("{} doesn't exists".format(path))
            return False

    def _string2datetime(self, name):
        '''converts string to datetime format'''
        assert len(name) == 13, 'directory name {} is not 13 characters'.format(name)
        #FIXME: an better way?
        d, t    = name.split('_')
        _d      = [d[:4], d[4:6], d[6:]]
        _t      = [t[:2], t[2:]]
        Y, M, D = [int(a) for a in _d]
        h, m    = [int(a) for a in _t]
        # print(Y, M, D, "_", h, m)
        return dt(year=Y,month=M,day=D,hour=h,minute=m)

    def _datetime2string(self, datetime):
        '''converts datetime format to string'''
        assert type(datetime) is dt, "Input should be in {} format".format(dt)
        return datetime.strftime("%Y%m%d_%H%M")

    def get_subdirs(self, path):
        if self.check_path_exists(path):
            return sorted(os.listdir(path))
        else:
            print("Path {} doesn't exists!".format(target_dir))
            return None

    def get_subdirs_of_datetime(self, datetime):
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root, dir_name)
        return self.get_subdirs(path)

    def get_rgb_path(self, datetime):
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root, dir_name)
        return os.path.join(path, rgb)
    
    def get_depth_path(self, datetime):
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root, dir_name)
        return os.path.join(path, depth)