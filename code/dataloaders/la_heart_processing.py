import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd

#从指定的文件夹中读取NRRD格式的医学图像数据和标签，并对这些数据进行预处理，然后将其保存为HDF5格式的文件。
output_size =[112, 112, 80]
#用于执行从NRRD到HDF5格式的转换
def covert_h5():
    listt = glob('../../data/LA/2018LA_Seg_Training Set/*/lgemri.nrrd')
    for item in tqdm(listt):
        #读取nrrd文件，返回图像数据和头信息
        image, img_header = nrrd.read(item)
        #读取标签文件，返回标签数据和头信息
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        #将标签数据中值为255的像素位置设置为1，其余位置设置为0
        #将标签数据转换为uint8类型
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        f = h5py.File(item.replace('lgemri.nrrd', 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()