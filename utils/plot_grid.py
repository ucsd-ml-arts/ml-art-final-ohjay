import os
import imageio
import argparse
import matplotlib.pyplot as plt

"""
Plot a bunch of images as a grid.
Usage: python3 plot_grid.py <images_dir> <nrows> <ncols> <out_path>
"""

def plot_grid(image_paths, nrows, ncols, out_path):
    fig, ax = plt.subplots(nrows, ncols, figsize=[ncols, nrows])
    if nrows == 1 and ncols == 1:
        ax = [[ax]]
    elif nrows == 1:
        ax = [[col for col in ax]]
    elif ncols == 1:
        ax = [[row] for row in ax]
    idx = 0
    for row in ax:
        for col in row:
            im = imageio.imread(image_paths[idx])
            col.imshow(im, cmap='gray')
            col.axis('off')
            idx += 1
    fig.savefig(out_path)
    print('[o] saved figure to %s' % out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('nrows', type=int)
    parser.add_argument('ncols', type=int)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    images_dir = args.images_dir
    nrows = args.nrows
    ncols = args.ncols
    out_path = args.out_path

    image_paths = [
        os.path.join(images_dir, fname) for fname in os.listdir(images_dir)]
    plot_grid(image_paths, nrows, ncols, out_path)
