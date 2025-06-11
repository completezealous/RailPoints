import numpy as np
from collections import Sequence
from typing import Optional, Union
import torch

class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      z (torch.Tensor): The z coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be
          smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        b = b.long()
        key = b << 48 | key
    return key

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    x=arr[:,0]
    print("x:",x)
    print("arr:",arr)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def interleave_bits(x, y, z):
    MASKS = [
        0x9249249249249249,
        0x318c6318c6318c63,
        0x00f00f00f00f00f0,
        0x0000ff000000ff00,
        0x00000000ffff0000
    ]
    SHIFTS = [2, 4, 8, 16, 32]

    def spread_bits(v):
        v &= 0x000000003fffffff
        for mask, shift in zip(MASKS, SHIFTS):
            v = (v | (v << shift)) & mask
        return v

    xx = spread_bits(x)
    yy = spread_bits(y)
    zz = spread_bits(z)

    # Combine the spread out bits of x, y and z to get the z-order index
    return xx | (yy << 1) | (zz << 2)

def z_order_hash(arr):
    assert arr.ndim == 2 and arr.shape[1] == 3
    x, y, z = arr.T  # Transpose to get separate arrays for x, y, and z
    # 将数组转化成uint64类型
    x = x.astype(np.uint64,copy=False)
    print("x:",x)
    y = y.astype(np.uint64)
    print("y:",y)
    z = z.astype(np.uint64)
    print("z:",z)
    return interleave_bits(x, y, z)


def z_order_index(coord):
    #discrete_coord = np.floor(coord / np.array(voxel_size))
    #discrete_coord = discrete_coord.astype(np.uint64, copy=False)
    return z_order_hash(coord)

def voxelize_old(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))#根据体素大小获取对应的体素坐标
    print(type(discrete_coord))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)#对体素坐标进行哈希编码

    idx_sort = np.argsort(key)#对哈希编码进行排序
    key_sort = key[idx_sort]#获取排序后的哈希编码
    _, count = np.unique(key_sort, return_counts=True)#获取哈希编码的唯一值和计数
    #print("count:",count.shape)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count#随机选择一个点
        idx_unique = idx_sort[idx_select]#获取随机选择的点的索引
        return idx_unique
    else:  # val mode
        idx_start = np.cumsum(np.insert(count, 0, 0)[0:-1])
        idx_select = idx_start# + np.random.randint(0, count.max(), count.size) % count
        # idx_unique = idx_sort[idx_select]
        sorted_idx = np.zeros(key.shape[0]).astype(np.int)
        sorted_idx[idx_start] = 1#获取每个哈希编码的起始索引
        sorted_idx = np.cumsum(sorted_idx) - 1
        idx_recon = np.zeros(key.shape[0]).astype(np.int)   #重建的索引数据
        idx_recon[idx_sort] = sorted_idx#获取重建的索引数据
        return idx_recon

def voxelize(coord, voxel_size=0.05, hash_type='z_order', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))#根据体素大小获取对应的体素坐标
    #discrete_coord = discrete_coord.astype(np.int64)  # 使用int64类型以允许位运算
    #discrete_coord=discrete_coord.long()
    if hash_type == 'z_order':
        #key = z_order_index(discrete_coord)
        key=xyz2key(torch.from_numpy(discrete_coord[:,0]),torch.from_numpy(discrete_coord[:,1]),torch.from_numpy(discrete_coord[:,2]),depth=16)
        #print("key:",key)
        #print("key.shape:",key.shape)
    elif hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)#对体素坐标进行哈希编码
        print("key:",key)
        print("key.shape:",key.shape)

    idx_sort = np.argsort(key)#对哈希编码进行排序
    key_sort = key[idx_sort]#获取排序后的哈希编码
    _, count = np.unique(key_sort, return_counts=True)#获取哈希编码的唯一值和计数
    #print("count:",count.shape)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count#随机选择一个点
        idx_unique = idx_sort[idx_select]#获取随机选择的点的索引
        return idx_unique
    else:  # val mode
        idx_start = np.cumsum(np.insert(count, 0, 0)[0:-1])
        idx_select = idx_start# + np.random.randint(0, count.max(), count.size) % count
        # idx_unique = idx_sort[idx_select]
        sorted_idx = np.zeros(key.shape[0]).astype(np.int)
        sorted_idx[idx_start] = 1#获取每个哈希编码的起始索引
        sorted_idx = np.cumsum(sorted_idx) - 1
        idx_recon = np.zeros(key.shape[0]).astype(np.int)   #重建的索引数据
        idx_recon[idx_sort] = sorted_idx#获取重建的索引数据
        return idx_recon
