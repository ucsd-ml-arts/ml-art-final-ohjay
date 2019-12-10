#!/usr/bin/env python

import os
import imageio
import argparse
import cv2
import numpy as np
from scipy.misc import imresize

def get_shape(images_dir):
    """Assumption: all images have the same shape."""
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        image = imageio.imread(image_path)
        return image.shape
    raise RuntimeError('[-] No files in `%s`.' % images_dir)

def write_video(images_dir, out_path, fps):
    out_height, out_width = get_shape(images_dir)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    size = (out_width, out_height)
    success = out.open(out_path, fourcc, fps, size, True)
    if not success:
        print('[-] Failed to open the video writer.')
        return

    frames_written = 0
    for image_name in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        image = imageio.imread(image_path)
        image = image[:, :, ::-1]  # RGB -> BGR
        if image.dtype in (np.float, np.float64):
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        out.write(image)
        frames_written += 1
        print(frames_written)
    out.release()
    print('[+] Finished writing %d frames to %s.' % (frames_written, out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str, help='directory containing images to write')
    parser.add_argument('--out_path', '-o', type=str, default='out.mov')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    write_video(args.images_dir, args.out_path, args.fps)
