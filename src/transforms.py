# Copyright (c) EEEM071, University of Surrey

import math
import random
import torch.nn.functional as F
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np
from scipy.ndimage import gaussian_filter


class Random2DTranslation:
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(
            round(self.height * 1.125)
        )
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class RandomErasing:
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    Origin: https://github.com/zhunzhong07/Random-Erasing
    """

    def __init__(
        self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                    img[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                    img[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                else:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                return img

        return img


class ColorAugmentation:
    """
    Randomly alter the intensities of RGB channels
    Reference:
    Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural Networks. NIPS 2012.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ]
        )
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

class RandomHorizontalFlip:
    """
    Randomly flip the image horizontally with a given probability
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        return torch.flip(tensor, dims=[-1])


class RandomVerticalFlip:
    """
    Randomly flip the image vertically with a given probability
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        return torch.flip(tensor, dims=[-2])
    
class GaussianBlur(object):
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        # Check if the random value is greater than the probability
        if np.random.random() > self.p:
            return img

        # Convert the input image to a numpy array
        img_np = np.array(img)

        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), 0)

        # Convert the blurred image back to a PIL Image and ensure the data type is uint8
        blurred_img_pil = Image.fromarray(blurred_img.astype(np.uint8))

        return blurred_img_pil


class GaussianBlur(object):
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        # Check if the random value is greater than the probability
        if np.random.random() > self.p:
            return img

        # Convert the input image to a numpy array
        img_np = np.array(img)

        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), 0)
        # Convert the blurred image back to a PIL Image
        blurred_img_pil = Image.fromarray(blurred_img)

        return blurred_img_pil


class ElasticTransform:
    def __init__(self, alpha=1, sigma=0.05, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img)
            shape = img_np.shape
            dx = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0,
            ) * self.alpha
            dy = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0,
            ) * self.alpha

            x, y, z = np.meshgrid(
                np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2])
            )
            indices = (
                np.reshape(y + dy, (-1, 1)),
                np.reshape(x + dx, (-1, 1)),
                np.reshape(z, (-1, 1)),
            )

            img_np = map_coordinates(img_np, indices, order=1).reshape(shape)
            img = Image.fromarray(img_np)
        return img



def build_transforms(
    height,
    width,
    random_erase=True,  # use random erasing for data augmentation
    color_jitter=True,  # randomly change the brightness, contrast and saturation
    color_aug=True,  # randomly alter the intensities of RGB channels
    horizontal_flip=True, #randomly flip the images horizontally
    vertical_flip=True, #randomly flip the images vertically
    gaussian_noise=True,
    gaussian_blur=True, 
    elastic_transform=True, 

    **kwargs
):
    # use imagenet mean and std as default
    # TODO: compute dataset-specific mean and std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

    # build train transformations
    transform_train = []
    transform_train += [Random2DTranslation(height, width)]
    transform_train += [T.RandomHorizontalFlip()]
    if color_jitter:
        transform_train += [
            T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]
    transform_train += [T.ToTensor()]
    if color_aug:
        transform_train.append(ColorAugmentation())
        transform_train.append(normalize)
    if random_erase:
        transform_train.append(RandomErasing())
    if horizontal_flip:
        transform_train.append(RandomHorizontalFlip())
    if vertical_flip:
        transform_train.append(RandomVerticalFlip())
    if gaussian_noise:
        transform_train.append(GaussianNoise())
    if gaussian_blur:
        transform_train.append(GaussianBlur())
    if elastic_transform:
        transform_train.append(ElasticTransform())
        transform_train.append(normalize)
    transform_train = T.Compose(transform_train)
    # build test transformations
    transform_test = T.Compose(
        [
            T.Resize((height, width)),
            T.ToTensor(),
            normalize,
        ]
    )

    return transform_train, transform_test
