PARENT_DIR=$(dirname $(pwd))
MODEL_DIR=$PARENT_DIR/models
DATASET=$(dirname $PARENT_DIR)/dataset
OUTPUT_DIR=$(dirname $PARENT_DIR)/output
SHARED=$(dirname $PARENT_DIR)/shared
OUTPUT_IMAGES=$OUTPUT_DIR/images/retinanet/retinanet-cuda
OUTPUT_FILES=$OUTPUT_DIR/files/retinanet/retinanet-cuda
CODE_DIR=$(pwd)/src
RETINANET=$(pwd)/retinanet-examples


docker run \
--gpus all \
-v $MODEL_DIR:/models \
-v $DATASET:/dataset \
-v $CODE_DIR:/src \
-v $OUTPUT_IMAGES:/output/images \
-v $OUTPUT_FILES:/output/files \
-v $SHARED:/src/shared \
-w /src \
--rm -it \
--ipc=host \
retinanet-cuda