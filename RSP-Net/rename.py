import os
import shutil

def zero_pad_number(number, width):
    number_str = str(number)
    zeros_to_add = width - len(number_str)
    if zeros_to_add > 0:
        number_str = '0' * zeros_to_add + number_str
    return number_str

def rename_files_in_folder(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1]
            new_file_name = zero_pad_number(count, 6) + file_extension
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)
            count += 1

folder_path = '/data/semanticRail/dataset/sequences/10/labels'  # 替换为实际的文件夹路径

if os.path.exists(folder_path) and os.path.isdir(folder_path):
    rename_files_in_folder(folder_path)
    print("文件重命名完成！")
else:
    print("文件夹路径无效或不存在！")
