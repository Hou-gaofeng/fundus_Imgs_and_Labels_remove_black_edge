from __future__ import division, print_function
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2


def convert(fname, label_EX, label_HE, label_MA, label_SE):
    img = Image.open(fname)
    img_EX = Image.open(label_EX)
    img_HE = Image.open(label_HE)
    img_MA = Image.open(label_MA)
    img_SE = Image.open(label_SE)
    debug = 0
    # blurred = img.filter(ImageFilter.BLUR) 这里学长用了blur后的，可能可以更均匀的识别背景？ 我没有很明白
    # ba = np.array(blurred)
    ba = np.array(img)
    h, w, _ = ba.shape
    if debug > 0:
        print("h=%d, w=%d" % (h, w))
    # 这里的1.2, 32, 5, 0.8都是后续可以调整的参数。 只是暂时觉得用这个来提取背景不错。
    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)
        foreground = (ba > max_bg + 5).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if debug > 0:
            print(foreground, left_max, right_max, bbox)
        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # 如果弄到的框小于原图的80%，很可能出bug了，就舍弃这个框。
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        if debug > 0:
            print
        bbox = square_bbox(img)

    cropped_img = img.crop(bbox)
    cropped_label_EX = img_EX.crop(bbox)
    cropped_label_HE= img_HE.crop(bbox)
    cropped_label_MA = img_MA.crop(bbox)
    cropped_label_SE = img_SE.crop(bbox)

    # resized_img = cropped_img.resize([crop_size, crop_size])
    # resized_label = cropped_label.resize([crop_size, crop_size])
    return cropped_img, cropped_label_EX, cropped_label_HE, cropped_label_MA, cropped_label_SE


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def main():
    img_dir = 'lesion_segmentation/valid/image'
    label_dir = 'lesion_segmentation/valid/label'
    img_save_dir = 'remove_black_multi/valid/img'
    label_save_dir = 'remove_black_multi/valid/label'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(label_save_dir + '/SE'):
        ex = label_save_dir + '/EX'
        he = label_save_dir + '/HE'
        ma = label_save_dir + '/MA'
        se = label_save_dir + '/SE'
        os.makedirs(ex)
        os.makedirs(he)
        os.makedirs(ma)
        os.makedirs(se)

    image_list = os.listdir(img_dir)
    count = 0
    for idx in range(len(image_list)):
        file_name = os.path.splitext(image_list[idx])[0]
        label_filename = file_name + '.tif'
        image_path = os.path.join(img_dir, image_list[idx])
        EX = 'EX/' + label_filename
        HE = 'HE/' + label_filename
        MA = 'MA/' + label_filename
        SE = 'SE/' + label_filename
        label_EX = os.path.join(label_dir, EX)
        label_HE = os.path.join(label_dir, HE)
        label_MA = os.path.join(label_dir, MA)
        label_SE = os.path.join(label_dir, SE)

        img, convert_EX, convert_HE, convert_MA, convert_SE = convert(image_path, label_EX, label_HE, label_MA, label_SE)

        # img.show()
        # label.show()
        image_name = os.path.join(img_save_dir, image_list[idx])

        label_name_EX = os.path.join(label_save_dir, EX)
        label_name_HE = os.path.join(label_save_dir, HE)
        label_name_MA = os.path.join(label_save_dir, MA)
        label_name_SE = os.path.join(label_save_dir, SE)

        # 保存图像
        img.save(image_name)
        convert_EX.save(label_name_EX)
        convert_HE.save(label_name_HE)
        convert_MA.save(label_name_MA)
        convert_SE.save(label_name_SE)
        count += 1

        # 图像融合
        # img = cv2.imread(image_name)
        # label = cv2.imread(label_name)
        # dst = cv2.addWeighted(img, 0.8, label, 0.1, 0)
        # cv2.imshow('dst', dst)
        # cv2.waitKey(0)
        # exit()
    print(count)


if __name__ == '__main__':
    main()
