#!/bin/bash

set -ex

# `dirname <NAME>`:获取文档所在目录对于执行目录的相对路径
# `realpath <FILE>`:获取当前文件的绝对路径
# 当前脚本的绝对路径
CUR_DIR=$(realpath $(dirname $0))
# 项目根目录
ROOT_DIR=$(realpath $CUR_DIR/..)
cd $ROOT_DIR

# 可执行文件
MAIN="$ROOT_DIR/bazel-bin/example/pagerank" # process name
# MPI进程数
WNUM=4
# OpenMP线程数
WCORES=4

# `KEY:=Value`:若变量KEY未定义则设置为值Value
# 输入文件
INPUT=${INPUT:="$ROOT_DIR/data/graph/v100_e2150_ua_c3.csv"}
# 输出路径
OUTPUT=${OUTPUT:="/tmp/pagerank"}
# 是否为有向图
IS_DIRECTED=${IS_DIRECTED:=false}
# PageRank算法参数-epsilon
EPS=${EPS:=0.0001}
# PageRank算法参数-阻尼系数
DAMPING=${DAMPING:=0.85}
# 迭代次数
ITERATIONS=${ITERATIONS:=100}

# param
PARAMS+=" --threads ${WCORES}"
PARAMS+=" --input ${INPUT} --output ${OUTPUT} --is_directed=${IS_DIRECTED}"
PARAMS+=" --iterations ${ITERATIONS} --eps ${EPS} --damping ${DAMPING}"

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
