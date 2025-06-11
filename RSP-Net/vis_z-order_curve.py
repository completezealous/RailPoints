import numpy as np
from collections import Sequence
from typing import Optional, Union
import matplotlib.pyplot as plt
from open3d import io
import torch
'''
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
'''
def xyz2key( x, y, z=0, depth=16):
    key = np.zeros_like(x)
    for i in range(depth):
        mask = 1 << i
        key = (
            key
            | ((x & mask) << (2 * i + 2))
            | ((y & mask) << (2 * i + 1))
            | ((z & mask) << (2 * i + 0))
        )
    return key

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)

    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


#pcd = io.read_point_cloud('/home/lk/Desktop/point_pcd/2023-10-28_pcd/101135-100m_new.pcd')
#points_1= np.asarray(pcd.points)  # 获取点云坐标
#points=(points_1).astype(np.int64)

# 创建一组随机点
np.random.seed(42)
points = np.random.randint(0, 256, size=(200, 2))
# 转换点到Z-order键
z_indices = np.array([xyz2key(x_val, y_val) for x_val, y_val in points])
# 计算每个点的Z-order索引
#z_indices = [xyz2key(point[0], point[1], point[2]) for point in points]

#计算hash索引
hash_indices=fnv_hash_vec(points)
# 根据计算出的Z-order索引对点进行排序
#sorted_indices = np.argsort(z_indices)

#根据计算出的hash索引对点进行排序
sorted_indices = np.argsort(hash_indices)
sorted_points = points[sorted_indices]

# 画出点和连接线
plt.scatter(sorted_points[:, 0], sorted_points[:, 1], c='blue')
for i in range(len(sorted_points) - 1):
    plt.plot(sorted_points[i:i+2, 0], sorted_points[i:i+2, 1], c='red')
# 可视化
#fig = plt.figure()
#ax= fig.add_subplot(111, projection='3d')
#sc=ax.scatter(x, y, z, c=keys, cmap='viridis')
#colors = plt.cm.jet((intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities)))

#ax.scatter(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2],c=z_indices, marker='o')
#ax.scatter(points_1[:, 0], points_1[:, 1], points_1[:, 2],marker='o',cmap='viridis')
#plt.scatter(x, y, z,c=keys, cmap='viridis')
#plt.colorbar(sc)
#plt.title("Points Coded with Z-order Curve")
#ax.set_xlabel("X Coordinate")
#ax.set_ylabel("Y Coordinate")
#ax.set_zlabel("Z Coordinate")
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Points connected by hash')
plt.grid(True)
plt.show()