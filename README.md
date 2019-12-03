# Final Project

Owen Jow, owen@eng.ucsd.edu

## Abstract Proposal

FIRST STEP: Write up a description (in the form of an abstract) of what you will revisit for your final project. This should be one paragraph clearly describing your concept and approach. What are your desired creative goals? How are you expanding on something we covered in the class? How will you present your work next Wednesday in the final project presentations?

## TODO

### Implementation

- Generate voxel objects using [this pretrained thing](https://github.com/zck119/3dgan-release). Stack them to create amalgamations of voxel objects (adds layer of ML "creativity"; chair + chair = chair? chair + desk = ?). Animate the construction, building everything layer-by-layer, from the ground up. Precompute the building sequence, although allow the users to add blocks in real-time and delete blocks in real-time. Can also do stylization in real-time as before.
- No need to generate LEGOs in particular. Too much time/work, doesn't add much w.r.t. artistic vision.

- Allow users to add more and more objects to the 3D scene, s.t. it starts simple but gets messier and messier. Style transfer can also be used to increase the sense of "mess."
- Loosen interactivity constraint; add functionality to automatically write out images and compose them into a video animation.
- The objects will be voxel objects. Use a GAN to generate voxel objects based on a distribution of `obj` models. See e.g. http://3dgan.csail.mit.edu.
- Animate the construction of the voxel objects.
- Add brief flashes of simplicity (i.e. records of the past) in the ever-growing clutter.
- Provide a button to delete objects which is essentially ineffectual; even if users apply the button, the "stuff" should grow exponentially.

### Conceptual Ideas

- To get started, the user can click a button to add things to the scene, not realizing at that point that it's only going to get more and more cluttered; nothing will ever really go away; it will be difficult to really clean things up.
- Symbolism:
  - **information clutter**
  - entropy, the unceasing arrow of time
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

Upload your project report (4 pages) as a pdf with your repository, following this template: [google docs](https://docs.google.com/document/d/133H59WZBmH6MlAgFSskFLMQITeIC5d9b2iuzsOfa4E8/edit?usp=sharing).

## Model/Data

Briefly describe the files that are included with your repository:
- trained models
- training data (or link to training data)

## Code

Your code for generating your project:
- Python: generative_code.py
- Jupyter notebooks: generative_code.ipynb

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- What you include here will very much depend on the format of your final project
  - image files (`.jpg`, `.png` or whatever else is appropriate)
  - 3d models
  - movie files (uploaded to youtube or vimeo due to github file size limits)
  - audio files
  - ... some other form

## Technical Notes

Any implementation details or notes we need to repeat your work. 
- Does this code require other pip packages, software, etc?
- Does it run on some other (non-datahub) platform? (CoLab, etc.)

## References

References to any papers, techniques, repositories you used:
- Papers
- Repositories
- Blog posts
