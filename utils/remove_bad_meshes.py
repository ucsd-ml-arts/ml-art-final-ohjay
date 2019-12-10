"""
Remove meshes without texture coordinates.
"""

__author__ = 'Owen Jow'
__email__  = 'owen@eng.ucsd.edu'

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_dir', type=str)
    args = parser.parse_args()

    for fname in os.listdir(args.mesh_dir):
        fpath = os.path.join(args.mesh_dir, fname)
        with open(fpath, 'r') as f:
            if not any([line.strip().startswith('vt ') for line in f.readlines()]):
                # no texture coordinates; remove
                os.remove(fpath)
                print('Removed `%s`.' % fpath)
