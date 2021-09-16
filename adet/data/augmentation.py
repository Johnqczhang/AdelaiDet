import random
from detectron2.data.transforms.augmentation import Augmentation

import cv2
import numpy as np
from fvcore.transforms import transform as T

from typing import List
from detectron2.data.transforms import (
    RandomCrop,
    ResizeTransform,
    ColorTransform,
    StandardAugInput
)


def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                # return T.CropTransform(0, 0, image_size[1], image_size[0])

    return T.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )


class RandomPaddedResize(Augmentation):
    """ Resize image to a fixed target size with padding around borders """

    def __init__(self, shapes) -> None:
        """
        Args:
            shapes (list[(int, int)]): A list of target sizes (h, w) to sample from.
        """
        super().__init__()
        assert all(len(shape) == 2 for shape in shapes), (
            "shapes must be a list of tuples of two values"
            f" Got {shapes}."
        )
        self._init(locals())

    def get_transform(self, image) -> T.TransformList:
        h, w = image.shape[:2]
        shape = np.random.permutation(self.shapes)[0]
        ratio = min(shape[0] / float(h), shape[1] / float(w))
        newh, neww = round(ratio * h), round(ratio * w)

        tfms = []
        tfms.append(ResizeTransform(h, w, newh, neww))

        padh = (shape[0] - newh) / 2  # height padding
        padw = (shape[1] - neww) / 2  # width padding
        if padh > 0 or padw > 0:
            t, b = round(padh - 0.1), round(padh + 0.1)
            l, r = round(padw - 0.1), round(padw + 0.1)
            tfms.append(PadTransform(l, t, r, b, neww, newh, pad_value=127.5))

        return T.TransformList(tfms)


class PadTransform(T.PadTransform):
    def apply_image(self, img, pad_value=None):
        if pad_value is None:
            pad_value = self.pad_value

        if img.ndim == 3:
            padding = ((self.y0, self.y1), (self.x0, self.x1), (0, 0))
        else:
            padding = ((self.y0, self.y1), (self.x0, self.x1))
        return np.pad(
            img,
            padding,
            mode="constant",
            constant_values=pad_value,
        )

    def apply_segmentation(self, segmentation):
        return self.apply_image(segmentation, pad_value=0.)


class RandomColorJitterHSV(Augmentation):
    """
    Randomly change the hue, saturation, and brightness of an RGB image.
    The input image is assumed to have 'BGR' channel order.
    """

    def __init__(self, hsv_factor=[0., 0., 0.], prob=0.5) -> None:
        """
        Args:
            hsv_factor (list[float]): HSV factor in the range of [0, 1).
        """
        super().__init__()
        if isinstance(hsv_factor, list):
            hsv_factor = np.array(hsv_factor, dtype=np.float32)
        self._init(locals())

    def get_transform(self, image) -> T.Transform:
        assert image.shape[-1] == 3, "RandomSaturation only works on RGB images"
        do = self._rand_range() < self.prob
        if do:
            hsv_factor = self._rand_range(-self.hsv_factor, self.hsv_factor, 3) + 1
            return JitterHSVTransform(hsv_factor)
        else:
            return T.NoOpTransform()


class JitterHSVTransform(ColorTransform):
    def __init__(self, hsv_factor) -> None:
        super(ColorTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        img = np.clip(img * self.hsv_factor, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img


class AugInputList(StandardAugInput):
    def __init__(self, images: List[np.ndarray]) -> None:
        self.images = images
        # let self.image pointing to the first image in the list for the callback mechanism
        self.image = self.images[0]

    def transform(self, tfm: T) -> None:
        self.images = [
            tfm.apply_image(image) for image in self.images
        ]
        # manually update self.image to avoid missing due to asynchronous data parallel
        self.image = self.images[0]
