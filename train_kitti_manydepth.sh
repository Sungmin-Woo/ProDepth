# CUDA_VISIBLE_DEVICES=4 \
DATA_PATH="<KITTI DATA PATH>"
MONO_DATA_PATH="<KITTI SINGLE-FRAME PRETRAINED PATH>"
model_name=$1
GPU_NUM=4
BS=3
PY_ARGS=${@:5}
PORT=$2

EXP_DIR=./log
LOG_DIR=$EXP_DIR/$model_name
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch --master_port=$PORT --nproc_per_node $GPU_NUM -m ProDepth.train \
    --dataset kitti \
    --data_path $DATA_PATH \
    --log_dir $EXP_DIR  \
    --model_name $model_name \
    --split eigen_zhou \
    --height 192 \
    --width 640 \
    --ddp \
    --batch_size $BS \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --encoder lite \
    --freeze_teacher_and_pose \
    --mono_weights_folder $MONO_DATA_PATH\


