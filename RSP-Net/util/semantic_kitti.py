import os
import random
import scipy
import glob
import numpy as np
import torch
import yaml
import pickle
from util.data_util import data_prepare
from util.instance_augmentation import instance_augmentation

#Elastic distortion
def elastic(x, gran, mag):  #函数的主要目的是对输入的数据x进行弹性模糊处理，gran是模糊的粒度，mag是模糊的程度
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3
    bb = (np.abs(x).max(0)//gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x + g(x) * mag


class SemanticKITTI(torch.utils.data.Dataset):
    def __init__(self, 
        data_path, 
        voxel_size=[0.05, 0.05, 0.05],
        split='train', 
        return_ref=True,  #是否返回参考坐标
        label_mapping="util/semantic-rail.yaml",
        aug=True,
        rotate_aug=True, 
        flip_aug=True,  #是否进行随机翻转
        scale_aug=True,  #是否进行随机缩放
        scale_params=[0.95, 1.05],  #缩放参数
        transform_aug=True,  #是否进行随机变换
        trans_std=[0.1, 0.1, 0.1], #变幻参数
        elastic_aug=False,  #是否进行弹性变形
        elastic_params=[[0.12, 0.4], [0.8, 3.2]],  #一个包含两个子列表的列表，表示弹性变形数据增强的参数，分别为粒度和程度。
        ignore_label=255,
        voxel_max=None, 
        xyz_norm=False, 
        pc_range=None, 
        use_tta=None,
        vote_num=4,
    ):
        super().__init__()
        self.num_classes = 5
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        thing_class=semkittiyaml['thing_class']
        class_weight=semkittiyaml['class_weight']
        thing_list=[cl for cl,ignored in thing_class.items() if ignored]
        self.return_ref = return_ref
        self.split = split
        self.aug = aug
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.scale_params = scale_params
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.data_path = data_path
        self.elastic_aug = elastic_aug
        self.elastic_gran, self.elastic_mag = elastic_params[0], elastic_params[1]
        self.use_tta = use_tta
        self.vote_num = vote_num
        instance_pkl_path = '/data/SemanticKITTI/'


        #if self.instance_aug:
        #    self.inst_aug=instance_augmentation(instance_pkl_path+'instance_path.pkl',thing_list,class_weight,
        #                                    random_flip=True,random_add=True,random_rotate=True,local_transformation=True)

        if split == 'train':
            splits = semkittiyaml['split']['train']
        elif split == 'val':
            splits = semkittiyaml['split']['valid']
        elif split == 'test':
            splits = semkittiyaml['split']['test']
        elif split == 'trainval':
            splits = semkittiyaml['split']['train'] + semkittiyaml['split']['valid']
        else:
            raise Exception('Split must be train/val/test')

        self.files = []
        for i_folder in splits:
            self.files += sorted(glob.glob(os.path.join(data_path, "sequences", str(i_folder).zfill(2), 'velodyne', "*.bin")))

        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)
        self.voxel_size = voxel_size

    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.nusc_infos)
        return len(self.files)

    def __getitem__(self, index):
        if self.use_tta:
            samples = []
            for i in range(self.vote_num):
                sample = tuple(self.get_single_sample(index, vote_idx=i))
                samples.append(sample)
            return tuple(samples)
        #elif self.aug:
        #    samples = []
        #    for i in range(self.vote_num):
        #        sample = tuple(self.get_single_sample(index, vote_idx=i))
        #        samples.append(sample)
        #    return tuple(samples)

        return self.get_single_sample(index)

    def get_single_sample(self, index, vote_idx=0):#获取单个样本

        file_path = self.files[index]

        raw_data = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
        annotated_data = np.fromfile(file_path.replace('velodyne', 'labels')[:-3] + 'label',
                                        dtype=np.uint32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        points = raw_data[:, :4]

        if self.split != 'test':
            #annotated_data[annotated_data == 0] = self.ignore_label + 1
            annotated_data = annotated_data
            labels_in = annotated_data.astype(np.uint8).reshape(-1)
        else:
            annotated_data = annotated_data
            labels_in = annotated_data.astype(np.uint8).reshape(-1)
            #labels_in = np.zeros(points.shape[0]).astype(np.uint8)

        # Augmentation
        # ==================================================
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            if self.use_tta:
                flip_type = vote_idx % 4
            else:
                flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]

        if self.scale_aug:
               moise_scale = np.random.uniform(0.9, 1.1)
               points[:, 0] *= moise_scale
               points[:, 1] *= moise_scale
        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            points[:, 0:3] += noise_translate
            
        if self.elastic_aug:
            points[:, 0:3] = elastic(points[:, 0:3], self.elastic_gran[0], self.elastic_mag[0])
            points[:, 0:3] = elastic(points[:, 0:3], self.elastic_gran[1], self.elastic_mag[1])


        # ==================================================

        feats = points#[N,4]原始点云数据
        #print("raw_points.shape",points.shape)
        #print("raw_point",points)
        xyz = points[:, :3]


        if self.pc_range is not None:
            xyz = np.clip(xyz, self.pc_range[0], self.pc_range[1]) #限制点云范围
            #print("clip_xyz.shape",xyz.shape)

        if self.split == 'train':
            coords, xyz, feats, labels = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)

            return coords, xyz, feats, labels #,point_feats,point_xyz#coords:体素坐标，xyz:体素中心坐标，feats:体素特征，labels:体素标签
        else:
            coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
            #if self.split == 'val':
            return coords, xyz, feats, labels, inds_reconstruct#point_feats,point_xyz #返回坐标、xyz、特征、标签和重建索引
            #elif self.split == 'test':
                #return coords, xyz, feats, labels, inds_reconstruct, self.files[index]#self.files[index]返回文件路径
            #    return coords, xyz, feats, labels, inds_reconstruct