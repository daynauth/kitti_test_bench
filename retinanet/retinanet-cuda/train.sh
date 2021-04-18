PARENT_DIR=$(dirname $(pwd))
MODEL_DIR=$PARENT_DIR/models
DATASET=$(dirname $PARENT_DIR)/dataset


docker run \
--gpus all \
-v $MODEL_DIR:/models \
-v $DATASET:/dataset \
--rm -it \
--ipc=host \
retinanet-cuda \
odtk train /models/retinanet-kitti-rn50fpn_fp.pth \
--fine-tune /models/retinanet_rn50fpn.pth \
--classes 8 --iters 10000 --val-iters 1000 --lr 0.0005 \
--resize 512 --jitter 480 640 --images /dataset/Kitti \
--annotations  /dataset/Kitti-Coco/train/labels.json \
--val-annotations /dataset/Kitti-Coco/validate/labels.json \
--full-precision