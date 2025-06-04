#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <concave|convex> <near|medium|far>"
    exit 1
fi

SHAPE=$1
DISTANCE=$2

case $DISTANCE in
    near) OBJ_X=-1; OBJ_Y=21 ;;
    medium) OBJ_X=0; OBJ_Y=22 ;;
    far) OBJ_X=1; OBJ_Y=23 ;;
    *) echo "Invalid distance: $DISTANCE"; exit 1 ;;
esac

case $SHAPE in
    convex) OBJ_THETA=155; OBJ_X=$((OBJ_X + 1)) ;;
    concave) OBJ_THETA=-25 ;;
    *) echo "Invalid shape: $SHAPE"; exit 1 ;;
esac

../build/whiskit_gui --dir_out "../output/${SHAPE}_${DISTANCE}" \
    --SAVE 1 --SAVE_VIDEO 1 --WHISKER_NAMES R \
    --DEBUG 0 --OBJ_CONVEX_HULL 0 --ACTIVE 1 \
    --OBJ_PATH "../data/object/convex-shape.obj" \
    --POSITION -5 -10 0 --ORIENTATION 0 15 0 \
    --OBJ_SCALE 55 --OBJ_SPEED 0 --OBJ_X $OBJ_X --OBJ_Y $OBJ_Y --OBJ_Z 3 --OBJ_THETA $OBJ_THETA \
    --HZ 1000 --SUBSTEPS 100 --WHISKER_LINKS_PER_MM 1 --PRINT 0 \
    --CX 2 --CY -4 --CZ 4 --CDIST 40 --CPITCH 270.001 --CYAW 0
