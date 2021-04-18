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
odtk train /models/retinanet-kitti-rn50fpn_full.pth \
--backbone ResNet50FPN \
--images /dataset/Kitti \
--annotations  /dataset/Kitti-Coco/train/labels.json \
--val-images /dataset/Kitti-Coco/validate/data \
--val-annotations /dataset/Kitti-Coco/validate/labels.json \
--classes 8 \
--full-precision \
--batch 8 \
--lr 0.00003