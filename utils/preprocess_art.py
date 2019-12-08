import os
import lycon
import imageio
import argparse

def resize_and_write(im, side, out_filepath):
    if im.shape[0] != side or im.shape[1] != side:
        # Resize to (side, side)
        im = lycon.resize(im, width=side, height=side,
            interpolation=lycon.Interpolation.CUBIC)
    imageio.imwrite(out_filepath, im)
    print('Wrote `%s`.' % out_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-sl', '--side_len', type=int, default=256)
    parser.add_argument('-ss', '--step_size', type=int, default=128)
    parser.add_argument('--init_rescale', type=float, default=1.0)
    parser.add_argument('--no_boundary', action='store_true')
    args = parser.parse_args()

    images_dir = args.images_dir
    output_dir = args.output_dir
    step = args.step_size
    side = args.side_len
    no_boundary = args.no_boundary
    init_rescale = args.init_rescale

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created output directory: %s.' % output_dir)

    total_ims_written = 0
    for filename in os.listdir(images_dir):
        if filename.endswith('.csv'):
            continue
        filepath = os.path.join(images_dir, filename)
        im = imageio.imread(filepath)
        h, w = im.shape[:2]

        if init_rescale != 1.0:
            h = int(h * init_rescale)
            w = int(w * init_rescale)
            im = lycon.resize(im, width=w, height=h,
                interpolation=lycon.Interpolation.CUBIC)

        y0_start = step if no_boundary else 0
        x0_start = step if no_boundary else 0
        y0_end = h - side - step if no_boundary else h - side
        x0_end = w - side - step if no_boundary else w - side

        ims_written = 0
        for y0 in range(y0_start, y0_end, step):
            for x0 in range(x0_start, x0_end, step):
                sub_im = im[y0:y0+side, x0:x0+side]
                out_filepath = os.path.join(output_dir,
                    str(ims_written).zfill(5) + '_' + filename)
                resize_and_write(sub_im, side, out_filepath)
                ims_written += 1

        if not no_boundary:
            # Final row
            for x0 in range(0, w - side, step):
                sub_im = im[-side:, x0:x0+side]
                out_filepath = os.path.join(output_dir,
                    str(ims_written).zfill(5) + '_' + filename)
                resize_and_write(sub_im, side, out_filepath)
                ims_written += 1

            # Final col
            for y0 in range(0, h - side, step):
                sub_im = im[y0:y0+side, -side:]
                out_filepath = os.path.join(output_dir,
                    str(ims_written).zfill(5) + '_' + filename)
                resize_and_write(sub_im, side, out_filepath)
                ims_written += 1

            # Final row, final col
            sub_im = im[-side:, -side:]
            out_filepath = os.path.join(output_dir,
                str(ims_written).zfill(5) + '_' + filename)
            resize_and_write(sub_im, side, out_filepath)
            ims_written += 1

        total_ims_written += ims_written
    print('Total number of images written: %d.' % total_ims_written)
