import numpy as np
import os
# 创建文件名和索引的字典
'''name_to_index = {
        22024: 0, 22945: 1, 22186: 2, 21896: 3, 22303: 4, 22871: 5, 23615: 6, 23093: 7,
        22456: 8, 23261: 9, 21337: 10, 22407: 11, 19506: 12, 22072: 13, 22214: 14, 19422: 15,
        22671: 16, 21510: 17, 20488: 18, 22994: 19, 23395: 20, 22457: 21, 20806: 22, 21615: 23,
        20929: 24, 22661: 25, 22970: 26, 22672: 27, 20554: 28, 21302: 29, 22758: 30, 21192: 31
        # 添加更多的文件名和索引
}'''
'''name_to_index = {
        22671: 16, 21510: 17, 20488: 18, 22994: 19, 23395: 20, 22457: 21, 20806: 22, 21615: 23,
        20929: 24, 22661: 25, 22970: 26, 22672: 27, 20554: 28, 21302: 29, 22758: 30, 21192: 31,
        22325: 32, 20386: 33, 20421: 34, 22594: 35, 22503: 36, 22772: 37, 22844: 38, 21762: 39,
        23233: 40, 22055: 41, 22017: 42, 22777: 43, 22684: 44, 25341: 45, 23155: 46, 20888: 47,
    # 添加更多的文件名和索引    08
}'''
name_to_index = {#共有60个索引
    8265: 0, 6308: 1, 6662: 2, 11783: 3, 6167: 4, 7152: 5, 6506: 6, 7541: 7,
    8706: 8, 8086: 9, 11992: 10, 7220: 11, 11886: 12, 9160: 13, 10059: 14, 7839: 15,
    10495: 16,8212: 17, 8978: 18, 8714: 19, 9146: 20, 22024: 21, 22303: 22, 22456: 23,
    22671: 24,21510: 25,21615: 26, 23353: 27, 19539: 28, 22261: 29, 20795: 30, 22195: 31,
    21482: 32, 22772: 33,23233: 34,22684: 35, 25341: 36, 19722: 37, 19732: 38, 19524: 39,
    20778: 40, 22093: 41,18466: 42, 22782: 43, 20758: 44, 18302: 45, 21096: 46, 19490: 47,
    23346: 48, 20694: 49,20439: 50, 22276: 51, 19471: 52, 18581: 53, 21485: 54, 21349: 55,
    21902: 56, 22151: 57,21861: 58, 19618: 59
}

def test_to_semantickitti(test_path, label_folder_path):
            # 处理语义标签和实例ID
    # Load point cloud data
    data = np.loadtxt(test_path)
    labels = data[:]

    sem_kitti_labels = np.zeros((len(labels), 2), dtype=np.uint16)
    sem_kitti_labels[:, 0] = labels
    sem_kitti_labels[:, 1] = 0#instances
    label_path = os.path.join(label_folder_path, os.path.basename(test_path)[:-4] + '.label')
    sem_kitti_labels.tofile(label_path,sep="",format="%u")

def test_to_semantickitti_batch(test_folder_path, label_folder_path):
    if not os.path.exists(label_folder_path):
        os.makedirs(label_folder_path)
    test_file_list = os.listdir(test_folder_path)
    for test_file_name in test_file_list:
        if test_file_name.endswith('.txt'):
            test_file_path = os.path.join(test_folder_path, test_file_name)
            test_to_semantickitti(test_file_path, label_folder_path)

def rename_file(label_folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(label_folder_path)

    # 循环处理每个文件
    for file_name in files:
        # 确保文件名至少有5个字符
        #if len(file_name) >= 5:
            # 提取文件名的最后5位数字
        last_five_digits = int(file_name[:-6])

        # 查找文件名在字典中的索引
        if last_five_digits in name_to_index:
            index = name_to_index[last_five_digits]

                # 构建新的文件名
            new_name = f"{index:06d}.label"

                # 构建新的文件路径
            new_file_path = os.path.join(label_folder_path, new_name)

                # 如果新文件名不与已存在的文件重复，就重命名文件
            if not os.path.exists(new_file_path):
                os.rename(os.path.join(label_folder_path, file_name), new_file_path)
            else:
                print(f"文件名 {new_name} 已存在，跳过重命名")

    print("重命名完成")
