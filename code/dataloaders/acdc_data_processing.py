import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
#将ACDC数据集中的3D图像和对应的标签数据转换为2D切片，并保存为HDF5格式的文件 文件名格式为<图像名称>_slice_<切片索引>.h5 +
# acdc数据集的预处理
#读取所有acdc图像数据集的路径，并按照顺序排序
slice_num = 0
mask_path = sorted(glob.glob("../../data/ACDC/image/*.nii.gz"))
#遍历每个图像路径
for case in mask_path:
    #读取图像文件
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    #根据图像路径生成对应的标签路径
    msk_path = case.replace("image", "label").replace(".nii.gz", "_gt.nii.gz")
    #检查对应的标签文件是否存在
    if os.path.exists(msk_path):
        print(msk_path)
        #读取标签文件
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)
        #对图像进行归一化处理
        image = (image - image.min()) / (image.max() - image.min())
        print(image.shape)
        image = image.astype(np.float32)
        #提取图像名称，用于生成HDF5文件名
        item = case.split("/")[-1].split(".")[0]
        #检查图像和标签的尺寸是否一致
        if image.shape != mask.shape:
            print("Error")
        print(item)
        #遍历图像的每个切片
        for slice_ind in range(image.shape[0]):
            # 创建HDF5文件以存储当前切片的图像和标签
            f = h5py.File(
                '../../data/ACDC/data/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            # 关闭HDF5文件
            f.close()
            # 增加切片计数
            slice_num += 1

            # 输出转换结果和总切片数
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))
