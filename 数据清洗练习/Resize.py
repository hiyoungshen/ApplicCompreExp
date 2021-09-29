# -*-coding = utf-8 -*-
"""
流程：
1.读取指定文件夹所有文件(必须都是图片)
2.进行resize，并存储在指定文件夹下
修改值：
path_read: 需要进行修改的图片存储的文件夹
path_write: 修改后的图片存储的文件夹,必须为空，会对图片重新编号00000-09999
target_size：[x, y] 修改后文件的尺寸
"""
import os
import cv2


if __name__ == "__main__":
    path_read = "./pic_save/"
    path_write = "./pic_resize/"
    target_size = [512, 512]
    image_list = [x for x in os.listdir(path_read)]
    for num, img in enumerate(image_list):
        print(num, img)
        image = cv2.imread(path_read+img, cv2.IMREAD_COLOR)
        # print(path_read+"/"+img)
        # new_image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC)
        try:
            new_image = cv2.resize(image, (target_size[0], target_size[1]))
        except:
            pass
        image_dir = path_write+str(num).zfill(5)+'.jpg'
        cv2.imwrite(image_dir, new_image)
