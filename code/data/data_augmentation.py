# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

datagen = image.ImageDataGenerator(fill_mode='wrap', zoom_range=[1, 1])
SAVE_PATH="/home/songruoning/paper_code/20190328_thyroid_classify/preprocess/chenzhou/data_augmentation/"
gen_data = datagen.flow_from_directory("/home/songruoning/paper_code/20190328_thyroid_classify/test_data/",
                                      batch_size=1,
                                      shuffle=False,
                                      save_to_dir=SAVE_PATH,
                                      save_prefix='gen',
                                       target_size=(224, 224))

# 生成9张图
for i in range(9):
   gen_data.next()

# 找到本地生成图，把9张图打印到同一张figure上
name_list = glob.glob(SAVE_PATH+'/*')
fig = plt.figure()
for i in range(9):
   img = Image.open(name_list[i])
   sub_img = fig.add_subplot(331 + i)
   sub_img.imshow(img)
plt.show()
