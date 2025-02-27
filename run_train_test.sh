#!/bin/bash
# set -x
# set -e

# conda python environment setup (manually)
# conda activate UString

PHASE=$1
GPUS=$2
DATA=$3
BATCH_SIZE=$4

LOG_DIR="./logs"

# create log dir
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p -m 777 $LOG_DIR
    echo "mkdir -p -m 777 ${LOG_DIR} done"
fi

# create log file
LOG="${LOG_DIR}/${PHASE}_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

OUT_DIR=output/UString/vgg16

MODEL_FILE=lib/final_model_a3d_vgg16.pth
# MODEL_FILE=lib/final_model_crash_vgg16.pth      
# MODEL_FILE=lib/final_model_dad_vgg16.pth

# experiments on DAD dataset
case ${PHASE} in
	train)
    	CUDA_VISIBLE_DEVICES=$GPUS python main.py \
      		--dataset $DATA \
      		--feature_name vgg16 \
      		--phase train \
      		--base_lr 0.0005 \
      		--batch_size $BATCH_SIZE \
      		--gpus $GPUS \
      		--output_dir $OUT_DIR
    	;;
  	test)
    	CUDA_VISIBLE_DEVICES=$GPUS python main.py \
      	--dataset $DATA \
      	--feature_name vgg16 \
      	--phase test \
      	--batch_size $BATCH_SIZE \
      	--gpus $GPUS \
      	--visualize \
      	--output_dir $OUT_DIR \
      	--model_file $MODEL_FILE
    	;;
  	*)
    echo "Invalid argument!"
    exit
    ;;
esac

