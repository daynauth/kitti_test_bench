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
odtk export /models/retinanet-kitti-rn50fpn.pth /models/retinanet-kitti.onnx \
--size 480 640 
--batch 1