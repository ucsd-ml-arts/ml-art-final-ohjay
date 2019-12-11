import os
import yaml
import random
import imageio
import argparse
import numpy as np
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Vec3
from pynput import keyboard
from direct.gui.OnscreenImage import OnscreenImage

loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'sync-video 0')
loadPrcFileData('', 'load-file-type p3assimp')
loadPrcFileData('', 'win-size {w} {h}'.format(w=600, h=450))

KEY_W = keyboard.KeyCode.from_char('w')
KEY_A = keyboard.KeyCode.from_char('a')
KEY_S = keyboard.KeyCode.from_char('s')
KEY_D = keyboard.KeyCode.from_char('d')
KEY_F = keyboard.KeyCode.from_char('f')
KEY_U = keyboard.KeyCode.from_char('u')


def get_files_with_extension(folder, ext):
    paths = []
    for fname in os.listdir(folder):
        if fname.endswith(ext):
            paths.append(os.path.join(folder, fname))
    return paths


class OutputWindow:
    def __init__(self, window_name):
        self.window_name = window_name

        # Store which keys are currently pressed.
        self.is_pressed = {
            'left':       False,
            'right':      False,
            'forward':    False,
            'backward':   False,
            'yaw-left':   False,
            'yaw-right':  False,
            'pitch-up':   False,
            'pitch-down': False,
        }
        self.take_snapshot = False
        self.signal_add_model = False
        self.signal_del_model = False

        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release)
        self.listener.start()

    def show_bgr_image(self, image, delay=1):
        # image should be in BGR format
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(delay)
        key &= 255
        if key == 27 or key == ord('q'):
            print('pressed ESC or q, exiting')
            return False
        return True  # the show goes on

    def on_key_press(self, key):
        if key == keyboard.Key.left:
            self.is_pressed['left'] = True
            self.is_pressed['right'] = False
        elif key == keyboard.Key.right:
            self.is_pressed['right'] = True
            self.is_pressed['left'] = False
        elif key == keyboard.Key.up:
            self.is_pressed['forward'] = True
            self.is_pressed['backward'] = False
        elif key == keyboard.Key.down:
            self.is_pressed['backward'] = True
            self.is_pressed['forward'] = False
        elif key == KEY_A:
            self.is_pressed['yaw-left'] = True
            self.is_pressed['yaw-right'] = False
        elif key == KEY_D:
            self.is_pressed['yaw-right'] = True
            self.is_pressed['yaw-left'] = False
        elif key == KEY_W:
            self.is_pressed['pitch-up'] = True
            self.is_pressed['pitch-down'] = False
        elif key == KEY_S:
            self.is_pressed['pitch-down'] = True
            self.is_pressed['pitch-up'] = False

    def on_key_release(self, key):
        if key == keyboard.Key.left:
            self.is_pressed['left'] = False
        elif key == keyboard.Key.right:
            self.is_pressed['right'] = False
        elif key == keyboard.Key.up:
            self.is_pressed['forward'] = False
        elif key == keyboard.Key.down:
            self.is_pressed['backward'] = False
        elif key == KEY_A:
            self.is_pressed['yaw-left'] = False
        elif key == KEY_D:
            self.is_pressed['yaw-right'] = False
        elif key == KEY_W:
            self.is_pressed['pitch-up'] = False
        elif key == KEY_S:
            self.is_pressed['pitch-down'] = False
        elif key == KEY_F:
            self.signal_add_model = True
        elif key == KEY_U:
            self.signal_del_model = True
        elif key == keyboard.Key.space:
            self.take_snapshot = True


class BeautyApp(ShowBase):
    def __init__(self, layout, background_path=None):
        ShowBase.__init__(self)

        # Disable the camera trackball controls.
        self.disableMouse()
        # Load the environment model.
        self.scene = self.loader.loadModel('assets/bedroom/bedroom.egg')
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(5, 5, 5)
        self.scene.setPos(0, 0, 0)

        self.models = []  # actual models
        self.model_velocities = {}  # {falling model idx: velocity}
        self.layout_h = layout.shape[0]
        self.layout_w = layout.shape[1]
        self.layout_size = self.layout_h * self.layout_w
        self.layout = layout.flatten()
        self.scene_x_scale = 54  # half of x-length of scene
        self.scene_y_scale = 72  # half of y-length of scene
        self.start_height = 39
        self.object_end_height = 13
        self.terminal_vel = 200  # hardcoded approximation (m)

        if background_path:
            # https://discourse.panda3d.org/t/background-image/5202/2
            self.bg = OnscreenImage(parent=render2d, image=background_path)
            self.camNode.getDisplayRegion(0).setSort(20)

        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

    def get_camera_image(self):
        """
        Returns the camera's image,
        which is of type uint8 and has values between 0 and 255.
        """
        tex = self.dr.getScreenshot()
        data = tex.getRamImageAs('RGB')
        image = np.frombuffer(data, np.uint8)  # change to `data.get_data()` if using Python 2
        image.shape = (tex.getYSize(), tex.getXSize(), 3)
        image = np.flipud(image)
        return image

    def add_model(self, mesh_path, texture_path, pos=None):
        self.models.append(self.loader.loadModel(mesh_path))
        self.models[-1].reparentTo(self.render)
        self.models[-1].setScale(0.09, 0.09, 0.09)
        self.models[-1].setTexture(
            self.loader.loadTexture(texture_path), 1)
        if pos is None:
            pos = self.sample_pos()
        self.models[-1].setPos(*pos)
        self.model_velocities[len(self.models) - 1] = 0
        print('added model %s at location %r' \
            % (mesh_path, self.models[-1].getPos()))

    def delete_random_model(self):
        num_models = len(self.models)
        if num_models > 0:
            model_idx = np.random.randint(0, num_models)
            self.models[model_idx].destroy()
            self.models[model_idx] = None
            del self.models[model_idx]
            if model_idx in self.model_velocities:
                del self.model_velocities[model_idx]
            print('deleted model %d' % model_idx)

    def update_falling_models(self, dt):
        # Perform position/velocity update for objects in motion.
        if len(self.model_velocities) == 0:
            return False
        for i, vel in self.model_velocities.items():
            # velocity update
            next_vel = vel - 9.8 * dt  # assume dt is in seconds
            next_vel = min(next_vel, self.terminal_vel)
            self.model_velocities[i] = next_vel
            # position update
            curr_z = self.models[i].getZ()
            updated_z = max(curr_z + next_vel * dt, self.object_end_height)
            self.models[i].setZ(updated_z)
        models_in_motion = list(self.model_velocities.keys())
        for i in models_in_motion:
            if self.models[i].getZ() == self.object_end_height:
                # object is no longer falling
                del self.model_velocities[i]
        return True

    @property
    def num_models_in_motion(self):
        """Returns the number of models in motion."""
        return len(self.model_velocities)

    def sample_pos(self):
        sample = np.random.choice(self.layout_size, 1, p=self.layout)
        y, x = np.unravel_index(sample, (self.layout_h, self.layout_w))

        # mask out already-sampled positions, re-normalize
        self.layout[y * self.layout_w + x] = 0
        layout_sum = np.sum(self.layout)
        if layout_sum == 0:
            # reset to uniform PDF
            self.layout = np.ones((self.layout_h, self.layout_w)) / self.layout_size
        else:
            self.layout /= layout_sum

        # ----------------------
        # convert pos2d to pos3d
        # ----------------------
        # remember that Panda3D uses a right-handed coordinate system
        # where x is right, y is forward, and z is up
        y = (float(y) / self.layout_h) * 2 - 1
        x = (float(x) / self.layout_w) * 2 - 1
        pos3d = (
            x * self.scene_x_scale,
            y * self.scene_y_scale,
            self.start_height
        )
        return pos3d

    def enforce_xy_bounds(self, x, y):
        """Take an X and a Y, cap values at scene X/Y limits."""
        x = min(max(x, -self.scene_x_scale), self.scene_x_scale)
        y = min(max(y, -self.scene_y_scale), self.scene_y_scale)
        return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--out_fps', type=int, default=30)  # for the offline version
    parser.add_argument('--layout_path', type=str)
    parser.add_argument('--background_path', type=str, default='assets/const.jpg')
    parser.add_argument('--no_autospawn', action='store_true')
    args = parser.parse_args()

    offline = args.offline
    out_dir = args.out_dir
    layout_path = args.layout_path
    background_path = args.background_path
    fps = args.out_fps
    no_autospawn = args.no_autospawn
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    mesh_dir = config['mesh_dir']
    texture_dir = config['texture_dir']
    layout_dir = config['layout_dir']

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('Created output directory at `%s`.' % out_dir)

    mesh_paths = get_files_with_extension(mesh_dir, '.obj')
    texture_paths = get_files_with_extension(texture_dir, ('.jpg', '.png'))

    if not layout_path:
        # select random layout path
        layout_paths = get_files_with_extension(layout_dir, ('.jpg', '.png'))
        layout_path = random.choice(layout_paths)
    layout = imageio.imread(layout_path, as_gray=True)
    # turn layout into PDF for sampling
    layout = np.squeeze(layout)
    layout /= np.sum(layout)  # normalize to [0, 1], sum to 1
    assert np.isclose(np.sum(layout), 1), \
        'probabilities sum to %f instead of 1' % np.sum(layout)

    app = BeautyApp(layout, background_path)
    window_name = 'Inexorable'
    if offline:
        frames = 1800
        output_window = None
    else:
        frames = 99999
        output_window = OutputWindow(window_name)

    num_objs_added = 0
    num_objs_to_add = 100
    obj_delete_throttle = 60

    pos_step = 0.7
    rot_step = 0.8
    start_time = time.time()
    if offline:
        obj_add_delay = (frames / fps) / num_objs_to_add
        prev_add_time = 0  # time of prev object drop
    else:
        obj_add_delay = 5  # of seconds until next object drop
        prev_add_time = start_time  # time of prev object drop
    init_add_delay = obj_add_delay

    # initial cam extrinsics
    cam_height = 22
    app.cam.setPos(0, 0, cam_height)
    app.cam.setHpr(0, 0, 0)
    app.graphicsEngine.renderFrame()  # for background

    image = None
    prev_time = 0 if offline else time.time()
    for t in range(frames):
        update = (t == 0)

        curr_time = t / fps if offline else time.time()
        update = update or app.update_falling_models(curr_time - prev_time)

        if not no_autospawn and num_objs_added < num_objs_to_add:
            if curr_time - prev_add_time >= obj_add_delay:
                # drop a random object onto the scene
                mesh_path = random.choice(mesh_paths)
                texture_path = random.choice(texture_paths)
                app.add_model(mesh_path, texture_path)
                # update add delay s.t. next object arrives sooner
                obj_add_delay -= init_add_delay / num_objs_to_add
                prev_add_time = curr_time

        if image is not None and (offline or output_window.take_snapshot):
            snapshot_path = 'frame%d.png' % (t - 1)
            snapshot_path = os.path.join(out_dir, snapshot_path)
            imageio.imwrite(snapshot_path, app.get_camera_image())
            print('Wrote `%s`.' % snapshot_path)
            if output_window:
                output_window.take_snapshot = False

        if offline:
            # spin around and look at scene as it builds up
            app.cam.setH(app.cam.getH() + rot_step)
            update = True
        else:
            # add model according to keypress
            if output_window.signal_add_model:
                # compute position in front of camera
                forward = app.render.getRelativeVector(app.cam, Vec3(0, 50, 0))
                curr_pos = app.cam.getPos()
                xa, ya = app.enforce_xy_bounds(curr_pos.x + forward.x,
                                               curr_pos.y + forward.y)
                pos_ahead = (xa, ya, app.object_end_height)

                mesh_path = random.choice(mesh_paths)
                texture_path = random.choice(texture_paths)
                app.add_model(mesh_path, texture_path, pos=pos_ahead)
                output_window.signal_add_model = False

            # delete model according to keypress
            if output_window.signal_del_model and t % obj_delete_throttle == 0:
                app.delete_random_model()
                output_window.signal_del_model = False

            # update yaw
            if output_window.is_pressed['yaw-left']:
                app.cam.setH(app.cam.getH() + rot_step)
                update = True
            elif output_window.is_pressed['yaw-right']:
                app.cam.setH(app.cam.getH() - rot_step)
                update = True

            # update pitch
            if output_window.is_pressed['pitch-up']:
                app.cam.setP(app.cam.getP() + rot_step)
                update = True
            elif output_window.is_pressed['pitch-down']:
                app.cam.setP(app.cam.getP() - rot_step)
                update = True

            # update pos: diagonal prep
            lr_pressed = output_window.is_pressed['left'] or \
                        output_window.is_pressed['right']
            ud_pressed = output_window.is_pressed['forward'] or \
                        output_window.is_pressed['backward']
            pos_step_adjusted = pos_step
            if lr_pressed and ud_pressed:
                pos_step_adjusted /= np.sqrt(2)

            # update pos
            if output_window.is_pressed['left']:
                right = app.render.getRelativeVector(app.cam, Vec3(1, 0, 0))
                curr_pos = app.cam.getPos()
                new_x = curr_pos.x - right.x * pos_step_adjusted
                new_y = curr_pos.y - right.y * pos_step_adjusted
                new_x, new_y = app.enforce_xy_bounds(new_x, new_y)
                app.cam.setPos(new_x, new_y, cam_height)
                update = True
            elif output_window.is_pressed['right']:
                right = app.render.getRelativeVector(app.cam, Vec3(1, 0, 0))
                curr_pos = app.cam.getPos()
                new_x = curr_pos.x + right.x * pos_step_adjusted
                new_y = curr_pos.y + right.y * pos_step_adjusted
                new_x, new_y = app.enforce_xy_bounds(new_x, new_y)
                app.cam.setPos(new_x, new_y, cam_height)
                update = True
            if output_window.is_pressed['forward']:
                forward = app.render.getRelativeVector(app.cam, Vec3(0, 1, 0))
                curr_pos = app.cam.getPos()
                new_x = curr_pos.x + forward.x * pos_step_adjusted
                new_y = curr_pos.y + forward.y * pos_step_adjusted
                new_x, new_y = app.enforce_xy_bounds(new_x, new_y)
                app.cam.setPos(new_x, new_y, cam_height)
                update = True
            elif output_window.is_pressed['backward']:
                forward = app.render.getRelativeVector(app.cam, Vec3(0, 1, 0))
                curr_pos = app.cam.getPos()
                new_x = curr_pos.x - forward.x * pos_step_adjusted
                new_y = curr_pos.y - forward.y * pos_step_adjusted
                new_x, new_y = app.enforce_xy_bounds(new_x, new_y)
                app.cam.setPos(new_x, new_y, cam_height)
                update = True

        if update:
            # render
            app.graphicsEngine.renderFrame()
            image = app.get_camera_image()
            image = image[:, :, ::-1]  # RGB -> BGR

        # show
        if not offline:
            if not output_window.show_bgr_image(image):
                break

        prev_time = curr_time

    if offline:
        # write frames to video
        from utils.write_video import write_video
        write_video(out_dir, 'out.mov', fps)

    end_time = time.time()
    print('average FPS: {}'.format(t / (end_time - start_time)))
