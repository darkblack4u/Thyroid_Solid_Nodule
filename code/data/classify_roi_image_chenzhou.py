import os

# In[2]:

"""
检查文件，参数（目录，后缀）
"""

def find_file(directory, extension):
    print("[INFO] directory: " + directory)
    target_files = []
    for target_file in os.listdir(directory):
        if target_file.endswith(extension):
            print("[INFO] " + extension + "_file: " + directory + "/" + target_file)
            target_files += [target_file]
    target_files.sort()
    return target_files

# 郴州ROI数据目录
folder_name = '/home/songruoning/data/preprocess/chenzhou/roi'

# 郴州数据输出目录
output_folder_name = '/home/songruoning/data/preprocess/chenzhou/classify_roi_image_chenzhou'
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)
    os.mkdir(output_folder_name + "/A1/")
    os.mkdir(output_folder_name + "/A2/")
    os.mkdir(output_folder_name + "/A3/")
    os.mkdir(output_folder_name + "/A4/")

# jpg文件列表
jpg_files = find_file(folder_name, '.jpg')
for jpg_file in jpg_files:
    target_folder=output_folder_name + "/" + jpg_file[0:2] + "/"
    if not os.path.exists(target_folder + jpg_file) or (os.path.exists(target_folder + jpg_file) and (os.path.getsize(target_folder + jpg_file) != os.path.getsize(folder_name + "/" + jpg_file))):
        open(target_folder + jpg_file, "wb").write(open(folder_name + "/" + jpg_file, "rb").read())
