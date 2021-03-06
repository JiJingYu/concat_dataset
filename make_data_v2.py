"""
该代码通过自行编写的H5Imageset类与pytorch中的ConcatDataset接口
有效利用了hdf5读取数据时直接与硬盘交互，无需载入整个数据集到内存中的优势，降低内存开销。
同时重载了python内置的__getitem__()方法，使得数据动态生成，无需独立保存数据，降低磁盘开销。
同时利用pytorch内置的ConcatDataset类，高效合并多组H5Imageset数据集，统一调用，统一索引。

该代码通常用于高维数据，如光场图像（4维），高光谱图像（3维），该类数据有数据量大，处理速度有限等特点。
传统的直接处理数据集、直接生成数据集、保存数据集的方法会使得数据量暴涨。例如ICVL数据集原始数据约30GB，
patch=64， stride=16分割之后，数据集会暴涨至500GB，给磁盘、IO和内存带来巨大压力。
用该代码可在不增加磁盘占用，不损失数据集IO时间的前提下，对数据集做有效的预处理，如按patch分割等，
同时可以大幅度降低内存占用。

由于空间有限，此处以少量RGB图像为例，简单展示demo用途。
made by 法师漂流
"""

import os
import h5py
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from data.dataset import H5Dataset, H5Imageset, ConcatDataset


def get_img():
    for img in os.listdir('./image'):
        if 'jpg' in img:
            yield img.split('.')[0], plt.imread('./image/'+img)


def make_data():
    img_g = get_img()
    f = h5py.File('img_data.h5', 'w')
    for img_name, img in img_g:
        d = f.create_dataset(name=img_name, data=img)
        ## TODO 这个数据会不会动态保存在内存中，如何关闭
    f.close()


def get_data():
    f = h5py.File('img_data.h5', 'r')
    print(list(f.keys()))
    print([f[key].shape for key in f.keys()])


def load_data():
    f = h5py.File('img_data.h5', 'r')
    dst_list = []
    for key in f.keys():
        tmp = H5Imageset(f[key], f[key], patch_size=32, stride=8)
        dst_list.append(tmp)
    dst = reduce(lambda x, y : x+y, dst_list)
    dst_2 = ConcatDataset(dst_list)
    print(dst_list)
    return dst_2


if __name__=="__main__":
    make_data()
    get_data()
    load_data()