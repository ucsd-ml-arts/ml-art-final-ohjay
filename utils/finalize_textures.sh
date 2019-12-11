# run from root directory (utils/..)
cd subjective-functions
for tex_path in ../synthesized_textures/*.png; do
    KERAS_BACKEND=tensorflow python3 synthesize.py -s $tex_path \
                                                   -ss 2 \
                                                   -o 2 \
                                                   --max-iter 200 \
                                                   --output-width 512 \
                                                   --output-height 512 \
                                                   --output-dir ../subjective_textures
done

# combine into one folder (synthesized_textures)
cd ..
idx=0
for fpath in subjective_textures/*/I0010_F0000.png; do
    fname=$(basename "$fpath")
    mv $fpath "synthesized_textures/I0010_F0000_${idx}.png"
    (( idx++ ))
done
for subdir in subjective_textures/*/; do
    for fpath in `ls $subdir*.png | sort -rV`; do
        fname=$(basename "$fpath")
        mv $fpath "synthesized_textures/${fname%.png}_${idx}.png"
        (( idx++ ))
        break
    done
done
rm -rf subjective_textures
