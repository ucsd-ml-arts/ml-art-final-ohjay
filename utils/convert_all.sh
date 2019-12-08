# Run vox2mesh on all voxel objects.
# Note: execute this script from the root directory.

VOXEL_DIR=$1
OUTPUT_MESH_DIR=$2

NUM_SHAPES_PER_MAT_FILE=150
for mat_path in $VOXEL_DIR/*.mat; do
    for i in $(seq 1 $NUM_SHAPES_PER_MAT_FILE); do
        # Postprocess the voxel object.
        cd 3dgan-release/visualization/python
        python postprocess.py $mat_path -t 0.1 -i $i -mc 2
        pp_mat_path="${mat_path%.mat}_postprocessed.mat"
        cd -
        # Perform voxel-to-mesh conversion.
        out_mesh_path="$OUTPUT_MESH_DIR/out_${mat_path%.mat}_$i.obj"
        python3 utils/vox2mesh.py $pp_mat_path -o $out_mesh_path
        # Assign texture coordinates.
        cd mesh-parameterization/build
        ./add-texcoords $out_mesh_path $out_mesh_path
        cd -
    done
done
