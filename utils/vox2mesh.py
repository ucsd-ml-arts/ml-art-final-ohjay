"""
Convert a voxel object to a mesh.
"""

__author__ = 'Owen Jow'
__email__  = 'owen@eng.ucsd.edu'

import argparse
import numpy as np
import scipy.io as sio
from mayavi import mlab
from skimage import measure

def vox2mesh(volume):
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0.0)
    mlab.triangular_mesh([vert[0] for vert in verts],
                         [vert[1] for vert in verts],
                         [vert[2] for vert in verts],
                         faces)
    mlab.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vox_filepath', type=str)
    args = parser.parse_args()

    if args.vox_filepath.endswith('.mat'):
        vox = sio.loadmat(args.vox_filepath)['voxels']
        # take first shape
        volume = vox[0, 0]
        # binarize
        volume[volume > 0.5] = 1
        volume[volume < 1] = 0
        # take largest connected component... TODO
        vox2mesh(volume)
    else:
        dot_idx = args.vox_filepath.rfind('.')
        print('Unsupported file extension: %s' % args.vox_filepath[dot_idx+1:])
