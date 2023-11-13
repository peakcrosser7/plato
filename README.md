# Plato(柏拉图)

[![Build Status](https://travis-ci.org/Tencent/plato.svg?branch=master)](https://travis-ci.org/Tencent/plato) [![docker-build workflow](https://github.com/Tencent/plato/workflows/docker-build/badge.svg)](https://github.com/Tencent/plato/actions?workflow=docker-build) 

**A framework for distributed graph computation and machine learning at wechat scale, for more details, see [柏拉图简介](doc/introduction.md) | [Plato Introduction](doc/introduction_en.md).**

Authors(In alphabetical order):  Benli Li, Conghui He, Donghai Yu, Pin Gao, Shijie Sun, Wenqiang Wu, Wanjing Wei, Xing Huang, Xiaogang Tu, Yangzihao Wang, Yongan Li.

Contact: plato@tencent.com

Special thanks to [Xiaowei Zhu](https://coolerzxw.github.io/) and many for their work [Gemini](https://coolerzxw.github.io/data/publications/gemini_osdi16.pdf)[1]. Several basic utility functions in Plato is derived from Gemini, the design principle of some dual-mode based algorithms in Plato is also heavily influenced by Gemini's dualmode-engine. Thanks to Ke Yang and many for their work [KnightKing](http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf)[2] which served as foundation of plato's [walk-engine](plato/engine/walk.hpp).

## Dependencies

To simplify installation, Plato currently downloads and builds most of its required dependencies by calling following commands. You should call it at least once before any build operations.

```bash
# install compile dependencies.
sudo ./docker/install-dependencies.sh
# download and build staticlly linked libraries.
./3rdtools.sh distclean && ./3rdtools.sh install
```

## Environment
Plato was developed and tested on x86_64 cluster and [Centos 7.0](https://www.centos.org/). Theoretically, it can be ported to other Linux distribution easily.

## Test && Build

```bash
./build.sh
```

## Run

### Local

```bash
./scripts/run_pagerank_local.sh
```

### Production

*Prerequisite:*

1. A cluster which can submit MPI programs([Hydra](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager) is a feasible solution).
2. An accessible [HDFS](https://hadoop.apache.org/) where Plato can find its input and put output on it.

A sample submit script was locate in [here](./scripts/run_pagerank.sh), modify it based on your cluster's environment and run.


```bash
./scripts/run_pagerank.sh
```

## Documents

* [支持算法列表](./doc/ALGOs.md)
* [集群资源配置建议](./doc/Resources.md) | [Notes on Resource Assignment](./doc/Resources_en.md)

## Reference

[1] Xiaowei Zhu, Wenguang Chen, Weimin Zheng, Xiaosong Ma. Gemini: A computation-centric distributed graph processing system. 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’16)

[2] Ke Yang, Mingxing Zhang, Kang Chen, Xiaosong Ma, Yang Bai, Yong Jiang. KnightKing: A Fast Distributed Graph Random Walk Engine. In ACM SIGOPS 27th Symposium on Operating Systems Principles (SOSP ’19)

## 大型分布式图计算系统的图算法开发  -  OSPP2023

在 Plato 系统上实现 PersonalizedPageRank（个性化pagerank）、TrustRank（信任指数）、BeliefPropagation（置信度传播）三个图算法。

### Personalized PageRank 算法
* 核心算法文件: `plato/algo/ppr/personalized_pagerank.hpp`  
* 算法 CLI 应用文件: `example/personalized_pagerank.cc`
* 算法运行脚本: `scripts/run_ppr_local.sh`
* 基于 PUSH-PULL 切换优化的算法版本: `example/pushpull_ppr.cc` 和 `scripts/run_pushpull_ppr_local.sh`
* 算法正确性验证: 参照 [Spark-GraphX](https://github.com/apache/spark/blob/master/graphx/src/main/scala/org/apache/spark/graphx/lib/PageRank.scala) 和 [Neo4j PageRank](https://neo4j.com/docs/graph-data-science/current/algorithms/page-rank/) 相关实现

### TrustRank 算法
* 核心算法文件: `plato/algo/trustrank/trustrank.hpp`  
* 算法 CLI 应用文件: `example/trustrank.cc`
* 算法运行脚本: `scripts/run_trustrank_local.sh`
* 算法正确性验证: 参照 [TrustRank 论文](https://dl.acm.org/doi/10.5555/1316689.1316740) 及 [bhaveshgawri/PageRank](https://github.com/bhaveshgawri/PageRank/blob/master/TrustRank.py) 相关实现

### Belief Propagation 算法
* 核心算法文件: `plato/algo/bp/belief_propagation.hpp`  
* 算法 CLI 应用文件: `example/belief_propagation.cc`
* 算法运行脚本: `scripts/run_bp_local.sh`
* 算法正确性验证: 参照 [HewlettPackard/sandpiper](https://github.com/HewlettPackard/sandpiper) 和 [mbforbes/py-factorgraph](https://github.com/mbforbes/py-factorgraph) 相关实现
