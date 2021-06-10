from __future__ import division, print_function
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2


def convert(fname, label_img):
    img = Image.open(fname)
    label_img = Image.open(label_img)
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
    cropped_label = label_img.crop(bbox)
    # resized_img = cropped_img.resize([crop_size, crop_size])
    # resized_label = cropped_label.resize([crop_size, crop_size])
    return cropped_img, cropped_label


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def main():
    img_dir = 'lesion_segmentation/test/image'
    label_dir = 'lesion_segmentation/test/label'
    img_save_dir = 'remove_black_multi/test/img'
    label_save_dir = 'remove_black_multi/test/label'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    image_list = os.listdir(img_dir)
    count = 0
    for idx in range(len(image_list)):
        file_name = os.path.splitext(image_list[idx])[0]
        label_filename = file_name + '.tif'
        image_path = os.path.join(img_dir, image_list[idx])
        label_path = os.path.join(label_dir, label_filename)
        img, label = convert(image_path, label_path)
        # img.show()
        # label.show()
        image_name = os.path.join(img_save_dir, image_list[idx])
        label_name = os.path.join(label_save_dir, label_filename)
        img.save(image_name)
        label.save(label_name)
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
