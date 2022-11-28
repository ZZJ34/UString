#!/bin/bash

# command input (video file)
VIDEO_PATH=$1
GPUS=0

# conda python environment setup (manually)
# conda activate UString

VIDEO_FILENAME="${VIDEO_PATH##*/}"  # ./demo/000821.mp4
DIR="${VIDEO_PATH%/*}"  # ./demo
VID="${VIDEO_FILENAME%.*}"  # 000821

FEAT_FILE="$DIR/$VID"_feature.npz
RESULT_FILE="$DIR/$VID"_result.npz
VIS_FILE="$DIR/$VID"_vis.avi

echo "Global info..."
echo "Video : $VIDEO_FILENAME"
echo "Dir : $DIR"

if [ ! -f "$RESULT_FILE" ]; then
    if [ ! -f "$FEAT_FILE" ]; then
        # feature extraction task
        echo "Run feature extraction..."
        
        CUDA_VISIBLE_DEVICES=$GPUS python demo.py \
            --video_file $VIDEO_PATH \
            --task extract_feature \
            --gpu_id $GPUS \
            --mmdetection lib/mmdetection
        

        echo "Saved in: $FEAT_FILE"
    fi

    # run inference
    # echo "Run accident inference..."
   
    # CUDA_VISIBLE_DEVICES=$GPUS python demo.py \
    #     --task inference \
    #     --feature_file $FEAT_FILE \
    #     --ckpt_file demo/final_model_ccd.pth \
    #     --gpu_id $GPUS
    
    # echo "Saved in: $RESULT_FILE"
fi

# visualize
# echo "Run result visualization..."
# CUDA_VISIBLE_DEVICES=$GPUS python demo.py \
#     --task visualize \
#     --video_file $VIDEO_PATH \
#     --result_file $RESULT_FILE \
#     --vis_file $VIS_FILE

# echo "Saved in: $VIS_FILE"

# conda deactivate