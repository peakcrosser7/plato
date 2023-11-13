#!/bin/bash

set -ex

CUR_DIR=$(realpath $(dirname $0))
ROOT_DIR=$(realpath $CUR_DIR/..)
cd $ROOT_DIR

MAIN="$ROOT_DIR/bazel-bin/example/belief_propagation" # process name
WNUM=4
WCORES=4

INPUT_FACTORS=${INPUT_FACTORS:="$ROOT_DIR/data/graph/factor_graph_2.csv"}
OUTPUT=${OUTPUT:="/tmp/bp"}
EPS=${EPS:=0.0001}
ITERATIONS=${ITERATIONS:=100}

# param
PARAMS+=" --threads ${WCORES}"
PARAMS+=" --input_factors ${INPUT_FACTORS} --output ${OUTPUT}"
PARAMS+=" --iterations ${ITERATIONS} --eps ${EPS}"

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
