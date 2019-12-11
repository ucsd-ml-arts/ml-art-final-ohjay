#
# Prepares assets (meshes, textures, layouts) for `renderloop.py`.
# Equivalent to components 1 through 4 from the README.
#

DOWNLOAD_MET_DATA=false
MET_DATA_DIR="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/van_gogh_art"
MET_PREPROCESSED_DATA_DIR="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/van_gogh_prep"
MSG_MODELS_DIR="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/msg_models"
RESISC_DATA_DIR="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/NWPU-RESISC45"

### [1] Voxel object generation

cd 3dgan-release
th main.lua -gpu 1 -class all -bs 50 -sample -ss 150
rm output/car_sample.mat
rm output/gun_sample.mat
cd -

### [2] Voxel/mesh conversion

./utils/convert_all.sh 3dgan-release/output processed_objs

### [3] Mesh stylization (texture generation)

if [ "$DOWNLOAD_MET_DATA" = true ]; then
    cd openaccess
    git lfs pull
    cd ../The-Metropolitan-Museum-of-Art-Image-Downloader
    python met_download.py --csv=../openaccess/MetObjects.csv --out=$MET_DATA_DIR --artist="Vincent van Gogh"
    rm $MET_DATA_DIR/piece_info.csv
    cd -
fi
python3 utils/preprocess_art.py $MET_DATA_DIR $MET_PREPROCESSED_DATA_DIR --no_boundary --init_rescale 0.6

cd BMSG-GAN
export SM_CHANNEL_TRAINING=$MET_PREPROCESSED_DATA_DIR
export SM_MODEL_DIR=$MSG_MODELS_DIR/exp_1
python3 sourcecode/train.py --depth=6 \
                            --latent_size=512 \
                            --num_epochs=730 \
                            --batch_size=5 \
                            --feedback_factor=1 \
                            --checkpoint_factor=10 \
                            --flip_augment=True \
                            --sample_dir=samples/exp_1 \
                            --model_dir=$MSG_MODELS_DIR/exp_1 \
                            --images_dir=$MET_PREPROCESSED_DATA_DIR

python3 sourcecode/generate_samples.py --generator_file=$MSG_MODELS_DIR/exp_1/GAN_GEN_730.pth \
                                       --latent_size=512 \
                                       --depth=6 \
                                       --out_depth=5 \
                                       --num_samples=300 \
                                       --out_dir=../synthesized_textures

cd ..
./utils/finalize_textures.sh

### [4] Scene layout design

cd sdae
python3 train.py --batch_size 32 \
                 --learning_rate 0.001 \
                 --num_epochs 5000 \
                 --model_class CVAE \
                 --dataset_key resisc \
                 --noise_type gs \
                 --gaussian_stdev 0.4 \
                 --save_path ./ckpt/cvae.pth \
                 --weight_decay 0.0000001 \
                 --dataset_path $RESISC_DATA_DIR

python3 generate_samples.py --model_class CVAE \
                            --restore_path ./ckpt/cvae.pth \
                            --num 30 \
                            --sample_h 256 \
                            --sample_w 256 \
                            --out_dir ../generated_layouts
