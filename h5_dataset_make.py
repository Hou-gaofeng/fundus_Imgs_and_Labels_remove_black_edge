import h5py
import os
import cv2

'''
PIL 读取的图片为RGB通道的Image格式
cv2 读取的图片为BGR通道的ndarray格式
skimage 读取的图片为RGB通道的格式
'''


def write_hdf5(arr, outfile):
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('image', data=arr, dtype=arr.dtype)


if __name__ == '__main__':
    h5_file_dir = 'single_re_black/valid_EX_h5dataset_single'
    if not os.path.exists(h5_file_dir):
        os.makedirs(h5_file_dir)

    original_imgs_train_file = './remove_black/valid/img'
    groundTruth_imgs_train_file = './remove_black/valid/label/EX/'

    label_list = []
    image_list = os.listdir(original_imgs_train_file)


    for idx in range(len(image_list)):
        file_name = os.path.splitext(image_list[idx])[0]
        label_filename = file_name + '.tif'
        image_path = os.path.join(original_imgs_train_file, image_list[idx])
        label_path = os.path.join(groundTruth_imgs_train_file, label_filename)

        im_b, im_g, im_r = cv2.split(cv2.imread(image_path))
        la_b, la_g, la_r = cv2.split(cv2.imread(label_path))

        write_dir = os.path.join(h5_file_dir, file_name + '.h5')
        im_g_normal = im_g / 255
        la_g_normal = (la_g / 255).astype(int)
        f = h5py.File(write_dir, 'w')
        f['image'] = im_g_normal
        f['label'] = la_g_normal
        f.close()
        print(file_name)






