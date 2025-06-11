import random
import torch
import numpy as np
from torch_scatter import scatter_mean
from util.voxelize import voxelize

def collate_fn_limit(batch, max_batch_points, logger):  #限制batch的点数
    coord, xyz, feat, label = list(zip(*batch))#将batch中的坐标、xyz、特征和标签分别组合成列表
    offset, count = [], 0

    new_coord, new_xyz, new_feat, new_label = [], [], [], []  #新的坐标，新的xyz，新的特征，新的标签
    k = 0
    for i, item in enumerate(xyz):#遍历xyz ,item为xyz中的每个元素

        count += item.shape[0]#计算每个体素均值的偏移
        if count > max_batch_points:
            break

        k += 1
        offset.append(count)#计算每个体素均值的偏移
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(label[i])

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in xyz])
        s_now = sum([x.shape[0] for x in new_xyz[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k]), torch.cat(new_label[:k]), torch.IntTensor(offset[:k])
    #返回新的坐标，新的xyz，新的特征，新的标签，新的偏移

def collation_fn_voxelmean(batch): #体素均值
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, xyz, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)#将batch中的坐标、xyz、特征、标签和重建索引分别组合成列表

    accmulate_points_num = 0
    offset = []
    for i in range(len(coords)):
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        offset.append(accmulate_points_num)#计算每个体素均值的偏移

    coords = torch.cat(coords)
    xyz = torch.cat(xyz)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    offset = torch.IntTensor(offset)
    inds_recons = torch.cat(inds_recons)

    return coords, xyz, feats, labels, offset, inds_recons

def collation_fn_voxelmean_tta(batch_list): #体素均值tta
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    samples = []
    batch_list = list(zip(*batch_list))

    for batch in batch_list:
        coords, xyz, feats, labels, inds_recons = list(zip(*batch))
        inds_recons = list(inds_recons)

        accmulate_points_num = 0
        offset = []
        for i in range(len(coords)):
            inds_recons[i] = accmulate_points_num + inds_recons[i]
            accmulate_points_num += coords[i].shape[0]
            offset.append(accmulate_points_num)

        coords = torch.cat(coords)
        xyz = torch.cat(xyz)
        feats = torch.cat(feats)
        labels = torch.cat(labels)
        offset = torch.IntTensor(offset)
        inds_recons = torch.cat(inds_recons)

        sample = (coords, xyz, feats, labels, offset, inds_recons)
        samples.append(sample)

    return samples

def data_prepare(coord, feat, label, split='train', voxel_size=np.array([0.1, 0.1, 0.1]), voxel_max=None, transform=None, xyz_norm=False):
    """
        准备数据函数，用于将原始坐标、特征和标签数据转换为适用于训练或测试的数据格式。
        参数：
        - coord: 原始坐标数据，形状为 (N, 3)，其中 N 是样本数量，3 是空间维度。
        - feat: 原始特征数据，形状为 (N, 4)，其中 N 是样本数量，4 是特征维度。
        - label: 原始标签数据，形状为 (N,)，其中 N 是样本数量。
        - split: 划分类型，可选值为 'train'（默认）或 'test'，表示数据是用于训练还是测试。
        - voxel_size: 体素大小，形状为 (D,)，其中 D 是空间维度。默认为 [0.1, 0.1, 0.1]。
        - voxel_max: 最大体素数，如果提供，将根据标签数量对数据进行裁剪。默认为 None。
        - transform: 坐标变换函数，如果提供，将对原始坐标进行变换。默认为 None。
        - xyz_norm: 是否对坐标进行归一化处理。默认为 False。

        返回值：
        - 如果 split 为 'train'，则返回以下元组：
          - coord_voxel: 转换后的体素坐标数据，形状为 (N,)。
          - coord: 转换后的原始坐标数据，形状为 (N, D)。
          - feat: 转换后的特征数据，形状为 (N, F)。
          - label: 转换后的标签数据，形状为 (N,)。
        - 如果 split 为 'test'，则返回以下元组：
          - coords_voxel: 转换后的体素坐标数据，形状为 (N,)。
          - coord: 转换后的原始坐标数据，形状为 (N, D)。
          - feat: 转换后的特征数据，形状为 (N, F)。
          - label: 转换后的标签数据，形状为 (N,)。
          - idx_recon: 重建的索引数据，形状为 (N,)。
        """
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)  #坐标变换函数
    coord_min = np.min(coord, 0) #坐标最小值
    # coord -= coord_min
    coord_norm = coord - coord_min#坐标归一化


    if split == 'train':
        uniq_idx = voxelize(coord_norm, voxel_size)  #体素化,获得非零坐标的索引
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))  # 根据体素大小获取对应的体素坐标
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]  #根据体素大小获取对应的坐标、特征和标签,数量小于原始点云数

        if voxel_max and label.shape[0] > voxel_max: #如果提供了最大体素数，将根据标签数量对数据进行裁剪
            init_idx = np.random.randint(label.shape[0])
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
            coord_voxel = coord_voxel[crop_idx]
    else:
       idx_recon = voxelize(coord_norm, voxel_size, mode=1)#对非零坐标进行体素化并获取重建的索引数据

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min

    # 将数据转换为张量类型
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    if split == 'train':
        #coord_norm = torch.FloatTensor(coord_norm)
        #idx_recon = torch.LongTensor(idx_recon)
        #coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        #coord_voxel=torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
        coord_voxel = torch.LongTensor(coord_voxel)  # 将体素坐标转换为长整型张量类型
        #coord = scatter_mean(coord, idx_recon, dim=0)
        #feat = scatter_mean(feat, idx_recon, dim=0)
        return coord_voxel, coord, feat, label #coord_voxel:体素坐标，coord:中心坐标，feat:特征，label:标签
    else:
        # 对非零坐标进行体素化并获取重建的索引数据
        coord_norm = torch.FloatTensor(coord_norm)
        idx_recon = torch.LongTensor(idx_recon)#重建的索引数据
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coords_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()#根据体素大小获取对应的体素坐标
        coord = scatter_mean(coord, idx_recon, dim=0)
        feat = scatter_mean(feat, idx_recon, dim=0)
        return coords_voxel, coord, feat, label, idx_recon
