# -*- coding: utf-8 -*-
from glob import glob
from pathlib import Path
import click
import logging
import numpy as np
from tqdm import tqdm
from skimage.util import view_as_blocks
from skimage.io import imread

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--input_size', type=click.IntRange(0), default=10_000)
@click.option('--output_size', type=click.IntRange(0), default=2_000)
def main(input_filepath: click.Path, output_filepath: click.Path, input_size: int, output_size: int):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # TRAIN IMAGES
    tr_input_path = Path(input_filepath).joinpath('train')
    tr_img_paths = glob(f'{tr_input_path}/*.jpeg')[:input_size]

    tr_output_path = Path(output_filepath).joinpath('train')
    tr_output_path.mkdir(exist_ok=True)

    for img_path in tqdm(tr_img_paths, 'Train Conversion'):
        img = convert_image(img_path)

        img_output_path = tr_output_path.joinpath(Path(img_path).stem)
        np.save(img_output_path, img)

    # TEST IMAGES
    test_input_path = Path(input_filepath).joinpath('test')
    test_img_paths = glob(f'{test_input_path}/*.jpeg')[:output_size]

    test_output_path = Path(output_filepath).joinpath('test')
    test_output_path.mkdir(exist_ok=True)
    
    for img_path in tqdm(test_img_paths, 'Test Conversion'):
        img = convert_image(img_path)

        img_output_path = test_output_path.joinpath(Path(img_path).stem)
        np.save(img_output_path, img)


def convert_image(img_path: str): 
    img = imread(img_path, as_gray=True)
    squares = view_as_blocks(img, block_shape=(8, 8))
    return squares

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
