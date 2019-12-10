# Final Project

Owen Jow, owen@eng.ucsd.edu

## Abstract Proposal

Here I revisit the 3D world of [my "generative visual" project](https://github.com/ohjay/in-pursuit-of-beauty), this time endeavoring to generate and place stylized objects in the scene according to an ML system's schizophrenic guidance. In a sentence, I aim to use ML to construct a stylized 3D scene (though without using 2D image stylization as in the previous project). I envision the scene growing into a mini metropolis, and intend for it to represent literally or seemingly inexorable processes such as entropy, virtual information clutter, and the proliferation of humanity's creations. The project involves several components, for which I provide brief descriptions [below](#project-components). Basically, I coerce some ML models to generate meshes, textures, and layouts to use for interactive rendering. I employ models and algorithms such as [Wu, Zhang et al.'s 3D-GAN](http://3dgan.csail.mit.edu/), [Karnewar et al.'s MSG-GAN](https://arxiv.org/abs/1903.06048) (similar to StyleGAN which was covered in class), convolutional denoising VAEs, marching cubes, mesh parameterization, and simple rasterization-based rendering. During the showcase, I mean to bring an interactive laptop demo in which users can walk around the scene as it is being constructed.

## Project Components

### ML-Based
- **[1] Voxel object generation.** I use [3D-GAN](https://github.com/zck119/3dgan-release) (Wu, Zhang et al.) to generate voxel objects.
- **[3] Mesh stylization.** I "stylize" the meshes by generating a texture ([1](https://github.com/akanimax/BMSG-GAN), [2](https://wxs.ca/research/multiscale-neural-synthesis)) to apply to each.
- **[4] Scene layout design.** I use ML to determine the placement of objects in the scene.

### Non-ML-Based
- **[2] Voxel/mesh conversion.** I convert the voxel objects to meshes using marching cubes, and UV map the meshes according to a cut-based parameterization method.
- **[5] Real-time scene construction.** I build up the scene using the stylized/generated objects in an animated fashion.
- **[6] Offline scene construction.** I also provide an option to write out the scene construction as a video.

## Extended Descriptions

### [1] Voxel object generation

- I generate objects for five classes of objects (car, chair, desk, gun, and sofa).
- Why voxels? It helps with layout generation. You can easily describe a set of voxel objects by a 2D top-down grid.

### [2] Voxel/mesh conversion

- I convert voxel objects to `.obj` files in order to reduce the number of meshes that Panda3D has to deal with (on the premise that I don't want to write a voxel engine).
- Unfortunately, the converted meshes are degenerate, and even a slew of online remeshing/repair software has failed to fix them to the point where many standard geometry processing operations (e.g. geodesic distances, harmonic map parameterizations) will actually work properly.

### [3] Mesh stylization

- I use GAN-based unconditional image generation to generate mesh textures. I choose to use [MSG-GAN](https://github.com/akanimax/BMSG-GAN) for its purported stability and because StyleGAN sounds like it might take longer to train.
- I also apply a synthesis method on top of this, such that I can generate an unlimited number of textures.

### [4] Scene layout design

- I train a generator as part of a convolutional VAE (CVAE) to create top-down layout sampling maps. These maps are grayscale and [0, 1]-normalized so that the value at each location can represent a probability. I use these maps in order to sample where to place each object. (Each pixel value describes the probability that we put each new object there.) I drop the objects from the sky as if they're coming from "the creator." An object will fall until it hits either the ground or an existing object, and in this way the scene is slowly built up. This stacking algorithm can create amalgamations of objects, which might contribute to a sense of ML "creativity" (e.g. sofa + sofa = ?).
- The full world is discretized as a 256x256x256 voxel grid. Users are prevented from leaving this area.
- The CVAE is trained on satellite imagery. Alternatively, it could be trained on any grayscale image data. For example, if it were trained on MNIST, then the scenes would build up like 3D numbers.
- The network is not trained for high-fidelity reconstruction. I do not regard this as important since (a) the maps are just used for sampling and (b) the application is primarily artistic, not reconstructive. It's the thought that counts. :)

### [5] Real-time scene construction

- I allow users to add objects to the 3D scene.
- I allow users to delete objects, but this functionality is essentially ineffectual, since the rate of growth is faster than the maximum rate of deletion.
- Users can orbit or walk around the scene as it grows.

### [6] Offline scene construction

- In this mode, the program will automatically compute and save an animation depicting the layer-by-layer construction of the scene.
- As part of the animation, I add brief flashes of simplicity (i.e. records of the past) in the ever-growing clutter.
- The scene begins as a natural environment with free-roaming pandas, but then the man-made objects take over.

## Project Report

You can find my project report [here](report/report.pdf).

## Model/Data

- You can download a pre-trained 3D-GAN model according to the instructions in [the repo](https://github.com/in-pursuit-of-beauty/3dgan-release).
- You can download a pre-trained MSG-GAN model from [this link](TODO).
  - This is the model which is used for generating artistic mesh textures. It was trained for 730 epochs on a dataset of Van Gogh paintings. I think it could stand to be trained for more.
  - Speaking of which, you can download and preprocess the Van Gogh dataset from [The Met Collection](https://www.metmuseum.org/art/collection) according to the instructions in the [stylization usage section](#3-mesh-stylization).
- You can download a pre-trained scene layout model from [this link](TODO).
  - This is the convolutional VAE model which is used to generate layout sampling maps. It was trained for X epochs on the RESISC45 satellite imagery dataset.
  - Speaking of which, you can download the RESISC45 dataset from [this link](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html).

## Code

- `renderloop.py`: The main file. Launches the rendering loop.
- `prepare_assets.sh`: Generates meshes, textures, and layouts in one fell swoop.
- `utils/vox2mesh.py`: Convert voxel grids to meshes and export as OBJs.
- `utils/preprocess_art.py`: Code for preprocessing/augmenting the art dataset.
- `utils/convert_all.sh`: Does postprocessing for all generated voxel objects.
- `utils/finalize_textures.sh`: Synthesizes additional textures.
- `utils/remove_bad_meshes.sh`: Removes meshes without texture coordinates.
- `utils/train_msg_gan.sh`: Convenience script for training the MSG-GAN (for use on cluster).
- `utils/write_video.py`: Takes a bunch of images and turns them into a video.
- `3dgan-release/main.lua`: Generates voxel objects.
- `3dgan-release/visualization/python/postprocess.py`: Postprocess voxel objects.
- `BMSG-GAN/sourcecode/train.py`: Train the art texture generator.
- `mesh-parameterization/src/main.cpp`: Add texture coordinates to OBJs.
- `mesh-parameterization/src/parameterize_mesh.cpp`: Do mesh parameterization.
- `The-Metropolitan-Museum-of-Art-Image-Downloader/met_download.py`: Download art data.
- `subjective-functions/synthesize.py`: Performs texture synthesis from input samples.
- `sdae/train.py`: Train the convolutional VAE for scene layout generation.
- `sdae/generate_samples.py`: Generate scene layouts using the trained VAE.

## Setup

```
git submodule update --init --recursive
cd mesh-parameterization
mkdir build
cd build
cmake ..
make
```

Download the [RESISC45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html).

## Quickstart

Edit the parameters at the beginning of `prepare_assets.sh`.
```
./prepare_assets.sh    # components [1]-[4]
python3 renderloop.py  # component [5]
```

## Usage

### [1] Voxel object generation

```
cd 3dgan-release
th main.lua -gpu 1 -class all -bs 50 -sample -ss 150
```

### [2] Voxel/mesh conversion

#### All

```
./utils/convert_all.sh 3dgan-release/output <mesh dir>
```

#### Individual

Postprocess the voxel object (binarize and keep only the largest connected component).
```
cd 3dgan-release/visualization/python
python postprocess.py <mat path> -t 0.1 -i 1 -mc 2
```

Perform the voxel-to-mesh conversion. The result will be written to an OBJ file.
```
python3 utils/vox2mesh.py <postprocessed mat path>
```

Assign texture coordinates based on a mesh parameterization.
```
cd mesh-parameterization/build
./add-texcoords <in.obj> <out.obj>
```

### [3] Mesh stylization

Download training data (public-domain Met art).
```
cd openaccess
git lfs pull
cd ../The-Metropolitan-Museum-of-Art-Image-Downloader
python met_download.py --csv=../openaccess/MetObjects.csv --out=<data dir> --artist="Vincent van Gogh"
rm <data dir>/piece_info.csv
```

Preprocess the training data.
```
python3 utils/preprocess_art.py <old data dir> <new data dir> --no_boundary --init_rescale 0.6
```

Train the MSG-GAN on the art data.
```
cd BMSG-GAN
export SM_CHANNEL_TRAINING=<data dir>
export SM_MODEL_DIR=<models dir>/exp_1
python3 sourcecode/train.py --depth=6 \
                            --latent_size=512 \
                            --num_epochs=500 \
                            --batch_size=5 \
                            --feedback_factor=1 \
                            --checkpoint_factor=20 \
                            --flip_augment=True \
                            --sample_dir=samples/exp_1 \
                            --model_dir=<models dir>/exp_1 \
                            --images_dir=<data dir>
```

Use the trained MSG-GAN to generate textures for meshes.
```
cd BMSG-GAN
python3 sourcecode/generate_samples.py --generator_file=<models dir>/exp_1/<checkpoint> \
                                       --latent_size=512 \
                                       --depth=6 \
                                       --out_depth=5 \
                                       --num_samples=300 \
                                       --out_dir=<texture dir>
```

Synthesize additional textures at a desired resolution.
```
./utils/finalize_textures.sh
```

### [4] Scene layout design

Train the layout design network. (Prerequisite: download the [RESISC45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html).)
```
cd sdae
python3 train.py --batch_size 32 \
                 --learning_rate 0.001 \
                 --num_epochs 500 \
                 --model_class CVAE \
                 --dataset_key resisc \
                 --noise_type gs \
                 --gaussian_stdev 0.4 \
                 --save_path ./ckpt/cvae.pth \
                 --weight_decay 0.0000001 \
                 --dataset_path <resisc data path>
```

Use the trained network to generate layouts.
```
cd sdae
python3 generate_samples.py --model_class CVAE \
                            --restore_path ./ckpt/cvae.pth \
                            --num 10 \
                            --sample_h 256 \
                            --sample_w 256 \
                            --out_dir <layouts dir>
```

### [5] Real-time scene construction

Set paths in the [config file](config.yaml), then run
```
python3 renderloop.py
```

### [6] Offline scene construction

Set paths in the [config file](config.yaml), then run
```
python3 renderloop.py --offline
```

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- What you include here will very much depend on the format of your final project
  - textures (drive)
  - meshes (drive)
  - layouts (drive)
  - snapshots of scene with everything (imgur)
  - video of scene (youtube)

## Technical Notes

To run 3D-GAN, you will need to install Torch (see [this](http://torch.ch/docs/getting-started.html) and maybe [this](https://github.com/nagadomi/waifu2x/issues/253#issuecomment-445448928)). For mesh visualization, you may want to install `mayavi` (this can be done via pip). To download the Met catalog, you will need [Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation).

I ran most of the computation-heavy code on a desktop computer with Ubuntu 18.04. You can also train the MSG-GAN on Jupyterhub, using e.g. `utils/train_msg_gan.sh` (I have confirmed that this works).

If you're using Python 2, you will need to edit one of the lines in [`renderloop.py`](renderloop.py). In `BeautyApp.get_camera_image`, change
```
image = np.frombuffer(data, np.uint8)
```
to
```
image = np.frombuffer(data.get_data(), np.uint8)
```

If you're running this on macOS, you should invoke `python3 renderloop.py` with sudo. Otherwise keyboard monitoring won't work due to security restrictions (see [here](https://pynput.readthedocs.io/en/latest/limitations.html#mac-osx)).

## Other Potential Directions

- Directly generate polygon meshes, e.g. based on [this paper](http://www.nobuyuki-umetani.com/publication/2017_siggatb_explore/2017_siggatb_ExploringGenerative3DShapes.pdf).
  - _Why didn't I do this?_ I wanted voxels for the obvious artificiality and for convenience during layout generation.
- Generate photorealistic materials, e.g. based on [this paper](https://keunhong.com/publications/photoshape/).
  - _Why didn't I do this?_ I wanted the objects to have a stylized quality to them, as opposed to being realistic.
- Generate voxel objects from sketches, e.g. based on [this project](https://github.com/maxorange/pix2vox).
  - _Why didn't I do this?_ Too much unnecessary overhead; doesn't really add to final product if non-interactive.
- Train the object generator on my own dataset, e.g. based on [this project](https://github.com/EdwardSmith1884/3D-IWGAN/tree/master/3D-Generation).
  - _Why didn't I do this?_ Not enough time.
- Generate higher-resolution voxel objects, e.g. based on [this project](https://github.com/EdwardSmith1884/Multi-View-Silhouette-and-Depth-Decomposition-for-High-Resolution-3D-Object-Representation).
  - _Why didn't I do this?_ Not enough time.
- More mesh cleanup, e.g. using [PyMesh](https://pymesh.readthedocs.io/en/latest/api_local_mesh_cleanup.html), CGAL, [`libigl`](https://github.com/libigl/libigl-examples/blob/master/skeleton-poser/clean.cpp), etc.
  - _Why didn't I do this?_ Not enough time, and unnecessary.
- Have the scene design network compute a new layout based on the current layout.
  - _Why didn't I do this?_ This would imply that at each step of the construction process, I would need to run the network.
  I wanted to precompute the sampling maps so that I could safely run the program on a laptop in real-time during the showcase.

## References

- Papers
  - [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](http://3dgan.csail.mit.edu)
  - [MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks](https://arxiv.org/pdf/1903.06048.pdf)
  - [High-Resolution Multi-Scale Neural Texture Synthesis](https://wxs.ca/research/multiscale-neural-synthesis)
- Repositories
  - [`3dgan-release`](https://github.com/zck119/3dgan-release)
  - [`BMSG-GAN`](https://github.com/akanimax/BMSG-GAN)
  - [`subjective-functions`](https://github.com/wxs/subjective-functions)
  - [`libigl`](https://github.com/libigl/libigl)
  - [`geometry-processing-parameterization`](https://github.com/alecjacobson/geometry-processing-parameterization)
  - [`mesh-parameterization`](https://github.com/ohjay/mesh-parameterization) (this is my repository, but...)
  - [The Met Image Downloader](https://github.com/trevorfiez/The-Metropolitan-Museum-of-Art-Image-Downloader)
  - [`metmuseum/openaccess`](https://github.com/metmuseum/openaccess)
  - [`sdae` (scene design autoencoder)](https://github.com/ohjay/sdae) [aka "stacked denoising autoencoder"]
- Other
  - [NumPy arrays from Panda3D textures - gist by Alex Lee](https://gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f)
  - ["Unconditional image generation" leaderboards](https://paperswithcode.com/task/image-generation)
  - [`scikit` marching cubes documentation](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner)
  - [`libigl` tutorial](https://libigl.github.io)
  - [The Met Collection](https://www.metmuseum.org/art/collection)
  - [RESISC45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)
