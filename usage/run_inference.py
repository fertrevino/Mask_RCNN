#!/usr/bin/env python3
"""
This file runs inference of the Mask-RCNN trained with the COCO dataset for a set of images.

Author: Fernando Trevino - fernando@yaneztrevino.com
"""
import os
import sys
import logging
import colorsys
from typing import Dict, List, Tuple

import numpy as np
import skimage.io
import matplotlib.axes
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
import mrcnn.model as modellib
from mrcnn import visualize

from inference_config import InferenceConfig, class_names
import argument_parser


def get_files_list(path: str, extension: str) -> List[str]:
    """Returns the images in the given path that have the desired extension"""
    return [os.path.join(path, file_) for file_ in os.listdir(path) if file_.endswith(extension)]


def load_image(path: str) -> np.ndarray:
    """Loads the image from the given path with assertion of being RGB"""
    loaded_image = skimage.io.imread(path)
    assert len(loaded_image.shape) == 3, f'Loaded image {path} is not RGB, please use RGB images'
    return loaded_image


def get_colors(class_names: List[str], class_ids: List[int]) -> List[Tuple[float]]:
    """Gets a list of equally distributed colors in the hsv color space for the number of classes available"""
    num_classes = len(class_names)
    hsv_tuples = [(x * 1.0 / num_classes, 0.5, 0.5) for x in range(num_classes)]
    rgb_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    return [rgb_tuples[id_] for id_ in class_ids]


def plot_detections(image: np.ndarray, detections: Dict, classes: List, ax: matplotlib.axes.Axes, colors: List) -> None:
    """Calls a customized version of the Mask-RCNN  visualization method to plot in the given axes"""
    visualize.display_instances(image,
                                detections['rois'],
                                detections['masks'],
                                detections['class_ids'],
                                classes,
                                detections['scores'],
                                ax=ax,
                                colors=colors)


def run_inference(images: List[str], model: modellib.MaskRCNN, out_path: str, human_read_classes: List[str]) -> None:
    """Runs the inference method on the list of images provided"""
    num_images = len(images)
    for count, path in enumerate(images[0:1], start=1):
        loaded_image = load_image(path)
        logging.info(f'Running detection for image {count}/{num_images}')
        detections = model.detect([loaded_image])[0]
        fig, ax = plt.subplots(1)
        plt.axis('off')
        colors = get_colors(human_read_classes, detections['class_ids'])
        plot_detections(loaded_image, detections, human_read_classes, ax, colors)
        save_path = os.path.join(out_path, os.path.basename(path))
        logging.info(f'Saving image: {save_path}')
        plt.savefig(save_path, dpi=90, bbox_inches='tight')


def main() -> None:
    """Main method to be called when this file is run"""
    arguments = argument_parser.parse()
    input_images = get_files_list(arguments.input_images, arguments.image_format)
    model = modellib.MaskRCNN(mode='inference', model_dir='', config=InferenceConfig())
    model.load_weights(arguments.weights, by_name=True)
    human_read_classes = list(class_names.values())
    run_inference(input_images, model, arguments.output_images, human_read_classes)


if __name__ == '__main__':
    main()
