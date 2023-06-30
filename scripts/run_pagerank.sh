#!/bin/bash

MAIN="./bazel-bin/example/pagerank" # process name

# MPI进程数
WNUM=4
# OpenMP线程数
WCORES=4

INPUT=${INPUT:='hdfs://cluster1/user/zhangsan/data/graph/raw_graph_10_9.csv'}
OUTPUT=${OUTPUT:='hdfs://cluster1/user/zhangsan/pagerank_raw_graph_10_9'}
# 是否为有向图
NOT_ADD_REVERSED_EDGE=${NOT_ADD_REVERSED_EDGE:=true}  # let plato auto add reversed edge or not

# 用于结点分区的参数
ALPHA=-1
# 是否根据结点入度划分
PART_BY_IN=false

# PageRank算法参数-epsilon
EPS=${EPS:=0.0001}
# PageRank算法参数-阻尼系数
DAMPING=${DAMPING:=0.85}
# 迭代次数
ITERATIONS=${ITERATIONS:=100}

export MPIRUN_CMD=${MPIRUN_CMD:='/opt/mpich-3.2.1/bin/mpiexec.hydra'}
export JAVA_HOME=${APP_JAVA_HOME:='/opt/jdk1.8.0_211'}
export HADOOP_HOME=${APP_HADOOP_HOME:='/opt/hadoop-2.7.4'}
export HADOOP_CONF_DIR="${HADOOP_HOME}/etc/hadoop"

PARAMS+=" --threads ${WCORES}"
PARAMS+=" --input ${INPUT} --output ${OUTPUT} --is_directed=${NOT_ADD_REVERSED_EDGE}"
PARAMS+=" --iterations ${ITERATIONS} --eps ${EPS} --damping ${DAMPING}"

# env for JAVA && HADOOP
export LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH}

# env for hadoop
export CLASSPATH=${HADOOP_HOME}/etc/hadoop:`find ${HADOOP_HOME}/share/hadoop/ | awk '{path=path":"$0}END{print path}'`
export LD_LIBRARY_PATH="${HADOOP_HOME}/lib/native":${LD_LIBRARY_PATH}

chmod 777 ./${MAIN}
${MPIRUN_CMD} -n ${WNUM} ./${MAIN} ${PARAMS}
# `$?`:上一个命令的退出状态码
# 以上一个命令的退出状态码退出
exit $?

