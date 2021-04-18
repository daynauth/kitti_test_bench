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
odtk infer /models/retinanet-kitti-rn50fpn_fp.pth \
--images /dataset/Kitti-Coco/validate/data \
--output /models/detection.json \
--batch 1