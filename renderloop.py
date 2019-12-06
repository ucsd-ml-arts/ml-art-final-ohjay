import imageio
import argparse
import numpy as np
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Vec3
from pynput import keyboard

loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'sync-video 0')
loadPrcFileData('', 'load-file-type p3assimp')
loadPrcFileData('', 'win-size {w} {h}'.format(w=600, h=450))

KEY_A = keyboard.KeyCode.from_char('a')
KEY_S = keyboard.KeyCode.from_char('s')


class OutputWindow:
    def __init__(self, window_name):
        self.window_name = window_name

        # Store which keys are currently pressed.
        self.is_pressed = {
            'left':      False,
            'right':     False,
            'forward':   False,
            'backward':  False,
            'yaw-left':  False,
            'yaw-right': False
        }
        self.take_snapshot = False

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
        elif key == KEY_S:
            self.is_pressed['yaw-right'] = True
            self.is_pressed['yaw-left'] = False

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
        elif key == KEY_S:
            self.is_pressed['yaw-right'] = False
        elif key == keyboard.Key.space:
            self.take_snapshot = True


class BeautyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Disable the camera trackball controls.
        self.disableMouse()
        # Load the environment model.
        self.scene = self.loader.loadModel('models/environment')
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

    def get_camera_image(self):
        """
        Returns the camera's image,
        which is of type uint8 and has values between 0 and 255.
        """
        tex = self.dr.getScreenshot()
        data = tex.getRamImageAs('RGB')
        image = np.frombuffer(data, np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), 3)
        image = np.flipud(image)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    app = BeautyApp()
    window_name = 'Inexorable'
    output_window = OutputWindow(window_name)

    frames = 99999
    pos_step = 0.2
    yaw_step = 0.5
    start_time = time.time()

    # initial cam extrinsics
    app.cam.setPos(42, -40, 3)
    app.cam.setHpr(18, 0, 0)

    image = None
    for t in range(frames):
        update = (t == 0)

        if image is not None and output_window.take_snapshot:
            snapshot_path = 'frame%d.png' % (t - 1)
            imageio.imwrite(snapshot_path, app.get_camera_image())
            print('Wrote `%s`.' % snapshot_path)
            output_window.take_snapshot = False

        # update yaw
        if output_window.is_pressed['yaw-left']:
            app.cam.setHpr(app.cam.getH() + yaw_step, 0, 0)
            update = True
        elif output_window.is_pressed['yaw-right']:
            app.cam.setHpr(app.cam.getH() - yaw_step, 0, 0)
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
            app.cam.setPos(new_x, new_y, 3)
            update = True
        elif output_window.is_pressed['right']:
            right = app.render.getRelativeVector(app.cam, Vec3(1, 0, 0))
            curr_pos = app.cam.getPos()
            new_x = curr_pos.x + right.x * pos_step_adjusted
            new_y = curr_pos.y + right.y * pos_step_adjusted
            app.cam.setPos(new_x, new_y, 3)
            update = True
        if output_window.is_pressed['forward']:
            forward = app.render.getRelativeVector(app.cam, Vec3(0, 1, 0))
            curr_pos = app.cam.getPos()
            new_x = curr_pos.x + forward.x * pos_step_adjusted
            new_y = curr_pos.y + forward.y * pos_step_adjusted
            app.cam.setPos(new_x, new_y, 3)
            update = True
        elif output_window.is_pressed['backward']:
            forward = app.render.getRelativeVector(app.cam, Vec3(0, 1, 0))
            curr_pos = app.cam.getPos()
            new_x = curr_pos.x - forward.x * pos_step_adjusted
            new_y = curr_pos.y - forward.y * pos_step_adjusted
            app.cam.setPos(new_x, new_y, 3)
            update = True

        if update:
            # render
            app.graphicsEngine.renderFrame()
            image = app.get_camera_image()
            image = image[:, :, ::-1]  # RGB -> BGR

        # show
        if not output_window.show_bgr_image(image):
            break

    end_time = time.time()
    print('average FPS: {}'.format(t / (end_time - start_time)))
