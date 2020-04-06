#!/usr/bin/env python3
"""
This file contains the argument parser used for running inference of Mask-RCNN.

Author: Fernando Trevino - fernando@yaneztrevino.com 
"""

import os
import argparse
import logging
from typing import Optional


def parse() -> argparse.Namespace:
    parser = _create_argument_parser()
    args = parser.parse_args()
    _check_arguments(args)
    return parser.parse_args()


def _check_arguments(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.input_images):
        logging.error(f'Given directory {args.input_images} does not exist. Please provide a valid argument.')
        raise NotADirectoryError

    _look_for_empty_dir(args.input_images, args.image_format)

    if not os.path.isdir(args.output_images):
        os.makedirs(args.output_images)
        logging.info(f'Given directory {args.output_images} does not exist and it has been created')


def _look_for_empty_dir(path: str, extension: Optional[str] = None) -> None:
    dir_files = [file_ for file_ in os.listdir(path) if file_.endswith(extension)] if extension else os.listdir(path)
    if not dir_files:
        logging.error(f'Directory {path} does not contain files with extension {extension}')
        raise ValueError


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run inference on a set of images.')
    parser.add_argument('--input_images', help='Path to folder with images to be used.', required=True)
    parser.add_argument('--output_images', help='Path to folder where output images will be stored', required=True)

    parser.add_argument('--image_format', help='Format extension of input images', default='.png')
    parser.add_argument('--weights', help='Path to the model weights.', default='../mask_rcnn_coco.h5')

    return parser
