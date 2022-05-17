import cv2
import numpy as np
import random


class ToGray(object):
    def __init__(self):
        pass

    def __call__(self, image_nc, image_art, image_pv, label):
        if len(image_nc.shape) == 3:
            image_nc = cv2.cvtColor(image_nc, cv2.COLOR_BGR2GRAY)
            image_nc = image_nc.reshape(image_nc.shape[0], image_nc.shape[1], 1)
            image_art = cv2.cvtColor(image_art, cv2.COLOR_BGR2GRAY)
            image_art = image_art.reshape(image_art.shape[0], image_art.shape[1], 1)
            image_pv = cv2.cvtColor(image_pv, cv2.COLOR_BGR2GRAY)
            image_pv = image_pv.reshape(image_pv.shape[0], image_pv.shape[1], 1)
        return image_nc, image_art, image_pv, label


class RondomFlip(object):
    def __init__(self):
        pass

    def __call__(self, image_nc, image_art, image_pv, label):
        degree = random.random()
        if degree <= 0.33:
            image_nc = cv2.flip(image_nc, 0)
            image_art = cv2.flip(image_art, 0)
            image_pv = cv2.flip(image_pv, 0)
        elif degree <= 0.66:
            image_nc = cv2.flip(image_nc, 1)
            image_art = cv2.flip(image_art, 1)
            image_pv = cv2.flip(image_pv, 1)
        return image_nc, image_art, image_pv, label


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image_nc, image_art, image_pv, label):
        angle = random.random() * self.angle
        angle = angle if random.random() < 0.5 else -angle
        h, w = image_nc.shape[0], image_nc.shape[1]
        scale = random.random() * 0.4 + 0.9
        matRotate = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        image_nc = cv2.warpAffine(image_nc, matRotate, (w, h))
        image_art = cv2.warpAffine(image_art, matRotate, (w, h))
        image_pv = cv2.warpAffine(image_pv, matRotate, (w, h))

        return image_nc, image_art, image_pv, label


class RandomCrop(object):
    def __init__(self, crop_h, crop_w):
        self.crop_h, self.crop_w = crop_h, crop_w

    def __call__(self, image_nc, image_art, image_pv, label):
        h, w = image_nc.shape[0], image_nc.shape[1]
        crop_x = int(random.random() * (w - self.crop_w))
        crop_y = int(random.random() * (h - self.crop_h))
        image_nc = image_nc[crop_y: crop_y + self.crop_h, crop_x: crop_x + self.crop_w]
        image_art = image_art[crop_y: crop_y + self.crop_h, crop_x: crop_x + self.crop_w]
        image_pv = image_pv[crop_y: crop_y + self.crop_h, crop_x: crop_x + self.crop_w]
        return image_nc, image_art, image_pv, label


class Blur(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image_nc, image_art, image_pv, label):
        if random.random() < self.degree:
            image_nc = cv2.blur(image_nc, (3, 3))
            image_art = cv2.blur(image_art, (3, 3))
            image_pv = cv2.blur(image_pv, (3, 3))
        return image_nc, image_art, image_pv, label


class Log(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image_nc, image_art, image_pv, label):
        if random.random() < self.degree:
            image_nc = np.log(1 + image_nc.astype(np.float32) / 255) * 255
            image_art = np.log(1 + image_art.astype(np.float32) / 255) * 255
            image_pv = np.log(1 + image_pv.astype(np.float32) / 255) * 255
        return image_nc.astype(np.uint8), image_art.astype(np.uint8), image_pv.astype(np.uint8), label


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image_nc, image_art, image_pv, label):
        image_nc = image_nc / 255
        image_art = image_art / 255
        image_pv = image_pv / 255

        with open(label, 'r') as f:
            label = f.readline()
        for idx, i in enumerate(label):
            if int(i) == 1:
                label = np.array(idx)
                break
        return image_nc.astype(np.float32),image_art.astype(np.float32),image_pv.astype(np.float32), label.astype(np.int64)


if __name__ == '__main__':
    transforms = [
        RandomCrop(2300, 2300),
        RondomFlip(),
        RandomRotate(15),
        Log(0.5),
        Blur(0.2),
        ToTensor(),
        ToGray()
    ]
