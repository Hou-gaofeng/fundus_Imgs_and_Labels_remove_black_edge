# 可视化mask和预测的label,将结果放在原图上显示
import cv2
import os


def mask2line(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 可以读取后缩放
def img_read(img, img_dir, scale=1):
    img = os.path.join(img_dir, img)
    img = cv2.imread(img)
    if img is not None:
        img = cv2.resize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)))
    return img


def main(img_label_dir, save_dir, thickness=3):

    label_dir_EX = img_label_dir + '/label/EX'
    label_dir_HE = img_label_dir + '/label/HE'
    label_dir_MA = img_label_dir + '/label/MA'
    label_dir_SE = img_label_dir + '/label/SE'

    img_dir = img_label_dir + '/img'
    image_list = os.listdir(img_dir)

    # 获取文件名称
    image_list = sorted([item.split(".")[0]
                         for item in image_list])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for name in image_list:
        print(name)
        img = img_dir + '/' + name + '.jpg'
        label_EX_file = label_dir_EX + '/' + name + '_EX.tif'
        label_HE_file = label_dir_HE + '/' + name + '_HE.tif'
        label_MA_file = label_dir_MA + '/' + name + '_MA.tif'
        label_SE_file = label_dir_SE + '/' + name + '_SE.tif'

        img = cv2.imread(img)
        label_EX = cv2.imread(label_EX_file)
        label_HE = cv2.imread(label_HE_file)
        label_MA = cv2.imread(label_MA_file)
        label_SE = cv2.imread(label_SE_file)

        # 绘图, cv2是bgr
        if label_EX is not None:
            contours_EX = mask2line(label_EX)
            cv2.drawContours(img, contours_EX, -1, (0, 0, 255), thickness)  # 红色 -1代表绘制所有的框
        if label_HE is not None:
            contours_HE = mask2line(label_HE)
            cv2.drawContours(img, contours_HE, -1, (0, 255, 0), thickness)  # 绿色
        if label_MA is not None:
            contours_MA = mask2line(label_MA)
            cv2.drawContours(img, contours_MA, -1, (255, 0, 0), thickness)  # 蓝色
        if label_SE is not None:
            contours_SE = mask2line(label_SE)
            cv2.drawContours(img, contours_SE, -1, (255, 0, 255), thickness)  # 紫色

        img_save = save_dir + '/' + name + '.png'
        cv2.imwrite(img_save, img)


if __name__ == '__main__':
    img_label_dir = './IDRiD/Segmentation/test'
    save_dir = './IDRiD/Segmentation/img_label_visualization/test'
    main(img_label_dir, save_dir)
