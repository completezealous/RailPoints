import copy
import numpy as np
import yaml
import os

import result_lable_rename as r

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d==0.9.0" to install open3d first.')




def _draw_bboxes(bbox3d, o3d_visualizer, points_colors, pcd, bbox_color, points_in_box_color, rot_axis, center_mode, mode):
    # Your code for drawing bounding boxes here
    pass  # Replace pass with your actual implementation
def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyzrgb'):
    vis.get_render_option().point_size = points_size  # set points size
    # if isinstance(points, torch.Tensor):
    #     points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    mode = "xyzrgb"
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    o3d.io.write_point_cloud("eg.pcd", pcd)
    return pcd, points_colors


class Visualizer(object):
    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path="pic",
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz',
                 filename=None):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.seg_num = 0
        self.filename = filename

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        if bbox_color is None:
            bbox_color = self.bbox_color
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def add_seg_mask(self, seg_mask_colors):
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_visualizer.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        _draw_points(
            seg_points, self.o3d_visualizer, self.points_size, mode='xyzrgb')

    def show(self, save_path=None):
        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return

def show_rawdata(label_path, pc_folder,show_filename=False):
    # 创建一个新的可视化器列表
    vis_ers = []
    with open('util/semantic-rail.yaml', 'r') as stream:
        CFG = yaml.safe_load(stream)
    # 遍历文件夹下的所有文件
    for filename in sorted(os.listdir(pc_folder)):
        # 检查文件是否为点云文件
        if filename.endswith('.bin'):
            # 读取标签文件
            with open(os.path.join(label_path, filename[:-4] + '.label'), 'r') as stream:
                raw_label = np.fromfile(stream, dtype=np.uint32).reshape((-1, 1))

            # 读取点云文件
            raw_data = np.fromfile(os.path.join(pc_folder, filename), dtype=np.float32).reshape((-1, 4))

            # 将标签数据映射到颜色
            color_dict = CFG["color_map"]
            learning_map = CFG["learning_map"]

            color_dict_mapped = dict()

            for cls in range(5):
                color_dict_mapped[cls] = color_dict[learning_map[cls]]
            color_dict_mapped[7] = [0, 128, 128]  # bicyclist
            color_dict_mapped[12] = [128, 128, 128]  # other-ground
            color_dict_mapped[14] = [212, 242, 231]  # fence
            color_dict_mapped[19] = [218, 165, 32]  # traffic-sign

            raw_label = raw_label & 0xFFFF  # delete high 16 digits binary
            raw_label = np.vectorize(learning_map.__getitem__)(raw_label).reshape((-1, 1))
            colors = []
            for label in raw_label[:, -1]:
                colors.append(color_dict_mapped[label][::-1])

            colors = np.vstack(colors)

            if raw_data.shape[1] == 4:
                points = raw_data[:, 0:3]
            else:
                points = raw_data[:, 3:6]
            color_points = np.concatenate((points, colors), axis=1)

            # 创建一个新的可视化器实例,
            #vis_er = Visualizer(color_points,filename=filename)

            if show_filename:
                filename_to_show = filename[:-4]
            else:
                filename_to_show=""

            vis_er = Visualizer(color_points, filename_to_show)
            vis_er.save_path = "pic_{}.ply".format(filename[:-4])
            vis_er.show()

            # 将可视化器添加到列表中
            vis_ers.append(vis_er)

    # 显示所有可视化结果
    for vis_er in vis_ers:
        vis_er.show()


if __name__ == "__main__":
    pc_path = '/home/lk/Desktop/sequences/09/velodyne/'
    #pc_path='/data/SemanticKITTI/dataset/sequences/17/velodyne/'
    label_path = '/data/semanticRail/dataset/sequences/09/labels/'
    #label_path='/home/lk/sequences_2/sequences/17/predictions/'
    #test_folder_path = "/home/lk/SphereFormer/test_result/test_result_20231224193841_mIoU_0.7858"
    #r.test_to_semantickitti_batch(test_folder_path, label_path)
    #if not r.rename_file(label_path):
    show_rawdata(label_path, pc_path,show_filename=False)

