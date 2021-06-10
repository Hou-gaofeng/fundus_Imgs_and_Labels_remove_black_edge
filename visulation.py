# 可视化mask和预测的label,将结果放在原图上显示
import cv2
import os


def mask2line(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def img_read(img, img_dir, scale=1):
    img = os.path.join(img_dir, img)
    img = cv2.imread(img)
    if img is not None:
        img = cv2.resize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)))
    return img


def main(img_label_dir, prediction_dir, prediction_file, save_dir):

    with open(prediction_file, 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for name in image_list:
        print(name)
        img = 'img/' + name + '.jpg'
        label = 'label/EX/' + name + '.tif'
        predict = name + '.tif'

        img = img_read(img, img_label_dir)
        label = img_read(label, img_label_dir)
        predict = img_read(predict, prediction_dir)
        if predict is None:
            continue

        contours_p = mask2line(predict)
        contours_l = mask2line(label)
        # 绘图
        cv2.drawContours(img, contours_p, -1, (0, 0, 255), 1)
        cv2.drawContours(img, contours_l, -1, (0, 255, 0), 1)
        img_save = save_dir + '/' + name + '.png'
        cv2.imwrite(img_save, img)


if __name__ == '__main__':
    img_label_dir = './remove_black/train'
    prediction_dir = './visulation/prediction_img'
    prediction_file = './visulation/val.list'
    save_dir = './visulation/img_label_prediction'
    main(img_label_dir, prediction_dir,prediction_file, save_dir)
