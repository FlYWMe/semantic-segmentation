import numpy as np
import cv2
import scipy.io as sio
def set_img_color(colors, background, img, gt,gt1, show255=True,showimg=0,ispred=0):
    print(len(colors))
    print(img.shape)
    print(gt.shape)
    if showimg==1:
        simg=1
    else:
        simg=0
    for i in range(simg, len(colors)):
        if i != background:
            # if i==0:
                # print(np.where(gt == i))
            img[np.where(gt == i)] = colors[i]

    if show255:
        img[np.where(gt == 255)] = 0
    if ispred==1:
    	img[np.where(gt1 == 255)] = 0
    return img
def show_prediction(colors, background, img, pred):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred)
    final = np.array(im)
    return final

def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    set_img_color(colors, background, im1, clean,gt,showimg=1,ispred=0)
    final = np.array(im1)

    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        print("pd: ",pd.shape)
        # pd = pd.permute(1, 2, 0)
        # pd = pd.cpu().numpy()
        im = np.array(img, np.uint8)
        # print(pd.shape)
        # pd=np.squeeze(pd)
        # print(pd.shape)

        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd,gt,ispred=1)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    print("1.")
    set_img_color(colors, background, im, gt,gt,ispred=0)
    print("2.")
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    # print(final.type)
    # print(final.shape)
    return final


def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1, 3)) * 255).tolist()[0])

    return colors


 