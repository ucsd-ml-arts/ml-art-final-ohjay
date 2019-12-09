# Final Project

Owen Jow, owen@eng.ucsd.edu

## Abstract Proposal

FIRST STEP: Write up a description (in the form of an abstract) of what you will revisit for your final project. This should be one paragraph clearly describing your concept and approach. What are your desired creative goals? How are you expanding on something we covered in the class? How will you present your work next Wednesday in the final project presentations?

## Project Components

### ML-Based
- **[1] Voxel object generation.** I use [3D-GAN](https://github.com/zck119/3dgan-release) (Wu, Zhang et al.) to generate voxel objects.
- **[3] Mesh stylization.** I "stylize" the meshes by generating a texture ([1](https://github.com/akanimax/BMSG-GAN), [2](https://wxs.ca/research/multiscale-neural-synthesis)) to apply to each.
- **[4] Scene layout design.** I use ML to determine the placement of objects in the scene.

### Non-ML-Based
- **[2] Voxel/mesh conversion.** I convert the voxel objects to meshes using marching cubes.
- **[5] Real-time scene construction.** I build up the scene using the stylized/generated objects in an animated fashion.
- **[6] Offline scene construction.** I also provide an option to write out the scene construction as a video.

## TODO

### Implementation

- Generate voxel objects using [this pretrained thing](https://github.com/zck119/3dgan-release). Stack them to create amalgamations of voxel objects (adds layer of ML "creativity"; chair + chair = chair? chair + desk = ?). Animate the construction, building everything layer-by-layer, from the ground up. Precompute the building sequence, although allow the users to add blocks in real-time and delete blocks in real-time. Can also do stylization in real-time as before.
- Allow users to add more and more objects to the 3D scene, s.t. it starts simple but gets messier and messier. Style transfer can also be used to increase the sense of "mess."
- Loosen interactivity constraint; add functionality to automatically write out images and compose them into a video animation.
- The objects will be voxel objects. Use a GAN to generate voxel objects based on a distribution of `obj` models. See e.g. http://3dgan.csail.mit.edu.
- Animate the construction of the voxel objects.
- Add brief flashes of simplicity (i.e. records of the past) in the ever-growing clutter.
- Provide a button to delete objects which is essentially ineffectual; even if users apply the button, the "stuff" should grow exponentially.
- Could add appropriate audio as well.
- Orbit/look/walk around scene and when you come back, there's a bunch more stuff than there was before.
- Try to stack the voxel objects like Tetris. Find a surface where there is an open space and where the current object can fit (or just stack them up however each individual layer of block falls).
- Start with nature scene with pandas roaming around. Add man-made objects. If you can do it in real-time, great. Maybe just precompute a sequence, and then can run it on the laptop, with a button to add pre-made ML objects to scene. But it has to look good; that's the point. It needs to have a nice aesthetic quality to it. Also, it needs to fulfill different aspects of the creativity metric.

- generate scene
- decide placement of objects based on 3d reconstructions of real scenes
  - Input: a floor plan (a 2D top-down floor plan)
  - Output: a floor plan with the new object (top-down space determined as voxels)
  - Because I have voxels; I can easily convert them into a 2D top-down voxel grid
  - and afterward I can also convert them to obj files to reduce number of meshes

- can determine material of objects at same time, based on color of the pixels in the floor plan?

- The world is a 1000x1000x1000 voxel grid. Can be filled. Cannot leave the area.

- A world of layers
- Discretize into squares of the size of the object, so that you know you won't overlap if possible
- RGBA image (current floor plan) -> RGBA image (new floor plan). Can train with RGBD images? (convert depth to inverse depth probability in [0, 1]; 1 is closer) [want the top-down to be like looking at an image; a beautiful image.] RGB represents desired color of object. A represents probability that we put the new object there. 1 is higher probability. Use A channel to sample where we put the next object.
- If entire floor plan is taken, use floor plan for next layer up. I guess objects will hang in the air in Panda3D, which is good.
- Make the network a small fully convolutional network so that I can run it on a laptop and it doesn't take too much memory or time. The actual quality is not extremely important since it is an artistic application and not a reconstructive one.

- Why voxels? Helps with layout generation. Can make into grid.

- triangle soup, degenerate meshes :(

- Generate textures: unconditional image generation (with a GAN?). Use [this](https://github.com/akanimax/BMSG-GAN). StyleGAN might take too long to train. Can also do texture synthesis on top of this, or separate from this (to generate an additional set of textures to use). Multiple texture generation methods.

### Conceptual Ideas

- To get started, the user can click a button to add things to the scene, not realizing at that point that it's only going to get more and more cluttered; nothing will ever really go away; it will be difficult to really clean things up.
- Symbolism:
  - **information clutter**
  - entropy, the unfaltering arrow of time
  - kipple from _Do Androids Dream of Electric Sheep?_
  - the internet, and how it's growing exponentially more cluttered and noisy as the number of people with access rises
  - how our lives are noisier now that we have electronic and network-based interference at all hours of the day
  - how ideas are everywhere, but it's hard to find the good ones
  - all the creators are trying to create, and we get more and more mess, more and more _things_; and ML is a creator too, one of many, and all of them are adding things at once; that's almost how it feels – or one creator makes something, and the others jump on the back of that
  - the LEGOs are supposed to represent something clearly artificial, something obviously man-made, connoting construction and man-made manufacture
- The style transfer represents how we make things and dress them up and call them beautiful, sometimes missing the content for the form (the aqua for the unda).
  - fulfills unda/aqua metaphor; style represents form, content represents essence; style can mask out the essence
  - maybe looks like a regular style transfer but there's something hidden in each image; as time goes on I make it so that the hidden thing is more and more evident by removing the style surrounding it (represents "getting down to the essence")
  - comment on postmodern aesthetic movement e.g. in Fowles's time and the time of "In Pursuit of Beauty"
- Cycles back from the cacophony and jumble of random things to a clean palette, a white slate. There are flashes of this, like memories, but the development is inevitable, inexorable, it must go on. Might depict how technology can get out of hand, and starts with a single push.
- Maybe having the program crash (can be "fake") is part of the app and the artistic vision. Alternatively, I can just say I considered this, but ultimately decided it would be too extreme.
- Issue: must appease creativity metric. Maybe ML decides how to add things to the scene, or creates the objects to add to the scene. This could be based on what it thinks would look good in the current version of the scene, according to typical scene composition – it could predict both the object and the position/orientation/scale of the object. (Solution: yes, at the very least, ML should be creating the objects to add to the scene.)
- Start with a natural-looking scene (even photorealistic, with later layers being composited in?), but then the LEGOs start to pile up.
- Only a few exceptions shine through, timeless, higher and taller than most.

## Project Report

Upload your project report (4 pages) as a pdf with your repository, following this template: [google docs](https://drive.google.com/open?id=1mgIxwX1VseLyeM9uPSv5GJQgRWNFqtBZ0GKE9d4Qxww).

## Model/Data

Briefly describe the files that are included with your repository:
- trained models
- training data (or link to training data)

## Code

- `renderloop.py`: The main file. Launches the rendering loop.
- `utils/vox2mesh.py`: Convert voxel grids to meshes and export as OBJs.
- `utils/preprocess_art.py`: Code for preprocessing/augmenting the art dataset.
- `3dgan-release/visualization/python/postprocess.py`: Postprocess voxel objects.
- `BMSG-GAN/sourcecode/train.py`: Train the art texture generator.
- `mesh-parameterization/src/main.cpp`: Add texture coordinates to OBJs.
- `mesh-parameterization/src/parameterize_mesh.cpp`: Do mesh parameterization.
- `The-Metropolitan-Museum-of-Art-Image-Downloader/met_download.py`: Download art data.

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
export SM_MODEL_DIR=models/exp_1
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
                                       --num_samples=300 \
                                       --out_dir=<texture dir>
```

Synthesize additional textures at a desired resolution.
```
cd subjective-functions
KERAS_BACKEND=tensorflow python3 synthesize.py -s <input_tex.jpg> \
                                               --output-width 512 \
                                               --output-height 512
```

### [4] Scene layout design

Train the layout design network. (Prerequisite: download the [RESISC45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html).)
```
cd sdae
python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --model_class MNISTSVAE \
    --dataset_key resisc \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --save_path ./ckpt/sdvae.pth \
    --weight_decay 0.0000001 \
    --vae_reconstruction_loss_type bce
```

### [5] Real-time scene construction

Set paths in the config file, then run
```
python3 renderloop.py
```

### [6] Offline scene construction

Set paths in the config file, then run
```
python3 renderloop.py --offline
```

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- What you include here will very much depend on the format of your final project
  - image files (`.jpg`, `.png` or whatever else is appropriate)
  - 3d models
  - movie files (uploaded to youtube or vimeo due to github file size limits)
  - audio files
  - ... some other form

## Technical Notes

To run 3D-GAN, you will need to install Torch (see [this](http://torch.ch/docs/getting-started.html) and maybe [this](https://github.com/nagadomi/waifu2x/issues/253#issuecomment-445448928)). For mesh visualization, you may want to install `mayavi` (this can be done via pip). To download the Met catalog, you will need [Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation).

You can train the MSG-GAN on Jupyterhub, using e.g. `utils/train_msg_gan.sh`.

Any implementation details or notes we need to repeat your work. 
- Does this code require other pip packages, software, etc?
- Does it run on some other (non-datahub) platform? (CoLab, etc.)

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
- Other
  - [NumPy arrays from Panda3D textures - gist by Alex Lee](https://gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f)
  - ["Unconditional image generation" leaderboards](https://paperswithcode.com/task/image-generation)
  - [`scikit` marching cubes documentation](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner)
  - [`libigl` tutorial](https://libigl.github.io)
  - [The Met Collection](https://www.metmuseum.org/art/collection)
  - [RESISC45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)
