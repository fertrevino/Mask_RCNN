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
    """Returns an argument parser for the supported inference arguments"""
    parser = _create_argument_parser()
    args = parser.parse_args()
    _check_arguments(args)
    return parser.parse_args()


def _create_argument_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the supported command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference on a set of images.')
    parser.add_argument('--input_images', help='Path to folder with images to be used.', required=True)
    parser.add_argument('--output_images', help='Path to folder where output images will be stored', required=True)

    parser.add_argument('--image_format', help='Format extension of input images', default='.png')
    parser.add_argument('--weights', help='Path to the model weights.', default='../mask_rcnn_coco.h5')

    return parser


def _check_arguments(args: argparse.Namespace) -> None:
    """Performs sanity checks to the provided input arguments"""
    _check_dir_exists(args.input_images)
    _check_file_exists(args.weights)
    _look_for_empty_dir(args.input_images, args.image_format)
    _make_dir_if_not_exist(args.output_images)


def _check_file_exists(path: str) -> None:
    """Checks that the specified path points to an existing file"""
    if not os.path.isfile(path):
        logging.error(f'Given file {path} does not exist. Please provide a valid argument.')
        raise FileNotFoundError


def _make_dir_if_not_exist(path: str) -> None:
    """Create the specified dir from the given path if it does not exist"""
    if not os.path.isdir(path):
        os.makedirs(path)
        logging.info(f'Given directory {path} does not exist and it has been created')


def _check_dir_exists(path: str) -> None:
    """Checks that the given path to a directory exists"""
    if not os.path.isdir(path):
        logging.error(f'Given directory {path} does not exist. Please provide a valid argument.')
        raise NotADirectoryError


def _look_for_empty_dir(path: str, extension: Optional[str] = None) -> None:
    """Checks that the given directory contains files with the expected extension"""
    dir_files = [file_ for file_ in os.listdir(path) if file_.endswith(extension)] if extension else os.listdir(path)
    if not dir_files:
        logging.error(f'Directory {path} does not contain files with extension {extension}')
        raise ValueError
