#!/bin/bash

set -ex

CUR_DIR=$(realpath $(dirname $0))
ROOT_DIR=$(realpath $CUR_DIR/..)
cd $ROOT_DIR

MAIN="$ROOT_DIR/bazel-bin/example/trustrank" # process name
WNUM=4
WCORES=4

INPUT_EDGES=${INPUT_EDGES:="$ROOT_DIR/data/graph/v100_e2150_ua_c3.csv"}
INPUT_GOODS=${INPUT_GOODS:="$ROOT_DIR/data/graph/v100_goods.csv"}
OUTPUT=${OUTPUT:="/tmp/trustrank"}
IS_DIRECTED=${IS_DIRECTED:=false}
EPS=${EPS:=0.0001}
DAMPING=${DAMPING:=0.85}
ITERATIONS=${ITERATIONS:=100}
SEED_NUM=${SEED_NUM:=100}
SELECT_METHOD=${SELECT_METHOD:=0}

# param
PARAMS+=" --threads ${WCORES}"
PARAMS+=" --input_edges ${INPUT_EDGES} --input_goods ${INPUT_GOODS} --output ${OUTPUT}"
PARAMS+=" --is_directed=${IS_DIRECTED} --iterations ${ITERATIONS} --eps ${EPS}"
PARAMS+=" --damping ${DAMPING} --seed_num ${SEED_NUM} --select_method ${SELECT_METHOD}"

# mpich
MPIRUN_CMD=${MPIRUN_CMD:="$ROOT_DIR/3rd/mpich/bin/mpiexec.hydra"}

# test
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$ROOT_DIR/3rd/hadoop2/lib

# output dir
mkdir -p $OUTPUT

# run
${MPIRUN_CMD} -n ${WNUM} ${MAIN} ${PARAMS}

echo ">>>>>>>>>>>output>>>>>>>>>>>>>>"
ls -lh $OUTPUT
