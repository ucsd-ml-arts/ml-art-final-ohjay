# Usage: ./train_msg_gan.sh <absolute path to data_dir>

# 0. Copy this file to the server.
# 1. Copy data to the server separately.
data_dir=$1

# 2. Clone the MSG-GAN repository.
git clone https://github.com/in-pursuit-of-beauty/BMSG-GAN.git
cd BMSG-GAN/sourcecode

export SM_CHANNEL_TRAINING=$data_dir
export SM_MODEL_DIR=models

# 3. Train.
python train.py --depth=7 \
                --latent_size=512 \
                --num_epochs=1001 \
                --batch_size=5 \
                --feedback_factor=1 \
                --checkpoint_factor=100 \
                --flip_augment=True \
                --sample_dir=samples \
                --model_dir=models \
                --images_dir=$data_dir
