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

def write_obj(path, verts, faces, normals):
    """Reference: https://stackoverflow.com/q/48844778."""
    faces = faces + 1
    with open(path, 'w') as f:
        for vert in verts:
            f.write('v {0} {1} {2}\n'.format(vert[0], vert[1], vert[2]))
        for normal in normals:
            f.write('vn {0} {1} {2}\n'.format(normal[0], normal[1], normal[2]))
        for face in faces:
            f.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(face[0], face[1], face[2]))
    print('Wrote `%s`.' % path)

def vox2mesh(volume, out_filepath, viz=False):
    verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0.0)
    write_obj(out_filepath, verts, faces, normals)
    if viz:
        mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                             faces)
        mlab.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('voxel_filepath', type=str)
    parser.add_argument('-o', '--out_filepath', type=str, default='out.obj')
    parser.add_argument('--viz', action='store_true')
    args = parser.parse_args()

    if args.voxel_filepath.endswith('.mat'):
        volume = sio.loadmat(args.voxel_filepath)['voxels']
        vox2mesh(volume, args.out_filepath, args.viz)
    else:
        dot_idx = args.voxel_filepath.rfind('.')
        print('Unsupported file extension: %s' % args.voxel_filepath[dot_idx+1:])
