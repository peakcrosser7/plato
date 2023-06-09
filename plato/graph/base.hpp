/*
  Tencent is pleased to support the open source community by making
  Plato available.
  Copyright (C) 2019 THL A29 Limited, a Tencent company.
  All rights reserved.

  Licensed under the BSD 3-Clause License (the "License"); you may
  not use this file except in compliance with the License. You may
  obtain a copy of the License at

  https://opensource.org/licenses/BSD-3-Clause

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" basis,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied. See the License for the specific language governing
  permissions and limitations under the License.

  See the AUTHORS file for names of contributors.
*/

#ifndef __PLATO_BASE_HPP__
#define __PLATO_BASE_HPP__

#include <cstdint>
#include <cstdlib>

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>

#include "omp.h"
#include "mpi.h"
#include "numa.h"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "libcuckoo/cuckoohash_map.hh"

#include "plato/util/backtrace.h"

// 声明命令行参数:线程数
DECLARE_int32(threads);

#define PAGESIZE (1 << 12)
#define HUGESIZE (1 << 20)
#define CHUNKSIZE (1 << 6)

#define MBYTES (1 << 20)
#define GBYTES (1UL << 30UL)
#define TBYTES (1UL << 40UL)

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

namespace plato {

using vid_t = std::uint32_t;
using eid_t = std::uint64_t;

struct empty_t {};

enum class edge_format_t { UNKNOWN = 0, CSV = 1 };

inline std::string edgeformat2name(edge_format_t format) {
    switch (format) {
        case edge_format_t::CSV:
            return "csv";
        default:
            return "unknown";
    }
}

inline edge_format_t name2edgeformat(const std::string& name) {
    if ("csv" == name || "CSV" == name) {
        return edge_format_t::CSV;
    }
    return edge_format_t::UNKNOWN;
}

// ******************************************************************************* //
// basic graph structure

/// @brief 边类型
template <typename EDATA_T, typename VID_T = vid_t>
struct edge_unit_t {
    /// @brief 起始结点
    VID_T src_;
    /// @brief 终止结点
    VID_T dst_;
    /// @brief 边数据
    EDATA_T edata_;

    template <typename Ar>
    void serialize(Ar& ar) {  
        // boost-style serialization when EDATA_T is non-trivial
        ar& src_& dst_& edata_;
    }
};  // __attribute__((packed));

/// @brief 边类型(无边数据特化)
template <typename VID_T>
struct edge_unit_t<empty_t, VID_T> {
    VID_T src_;
    union {
        VID_T dst_;
        empty_t edata_;
    };
};  // __attribute__((packed));

/// @brief 邻接表元素类型
/// @tparam EDATA_T 边数据类型
template <typename EDATA_T>
struct adj_unit_t {
    /// @brief (边的源/终)邻结点ID
    vid_t neighbour_;
    /// @brief 边数据
    EDATA_T edata_;

    template <typename Ar>
    void serialize(Ar& ar) {  
        // boost-style serialization when EDATA_T is non-trivial
        ar& neighbour_& edata_;
    }
};  // __attribute__((packed));

template <>
struct adj_unit_t<empty_t> {
    union {
        vid_t neighbour_;
        empty_t edata_;
    };

    template <typename Ar>
    void serialize(Ar& ar) {
        ar& neighbour_;
    }
};  // __attribute__((packed));

/// @brief 结点邻接表
template <typename EDATA_T>
struct adj_unit_list_t {
    /// @brief 起点
    adj_unit_t<EDATA_T>* begin_;
    /// @brief 终点
    adj_unit_t<EDATA_T>* end_;

    adj_unit_list_t(void) : begin_(nullptr), end_(nullptr) {}

    adj_unit_list_t(adj_unit_t<EDATA_T>* begin, adj_unit_t<EDATA_T>* end)
        : begin_(begin), end_(end) {}
};

template <typename VDATA_T, typename = typename std::enable_if<
                                sizeof(VDATA_T) != 0, std::true_type>::type>
struct vertex_unit_t {
    vid_t vid_;
    VDATA_T vdata_;

    template <typename Ar>
    void serialize(
        Ar& ar) {  // boost-style serialization when VDATA_T is non-trivial
        ar& vid_& vdata_;
    }
};  // __attribute__((packed));

// ******************************************************************************* //

// user should keep this struct alive during whole process lifetime
/// @brief 集群信息
struct cluster_info_t {
    /// @brief 分区总数(MPI进程总数)
    int partitions_;
    /// @brief 分区ID(进程ID)
    int partition_id_;

    /// @brief 线程总数
    int threads_;
    /// @brief NUMA节点数
    int sockets_;

    /// @brief 获取集群信息单例
    static cluster_info_t& get_instance(void) {
        static cluster_info_t instance;
        return instance;
    }

    /// @brief 集群初始化
    void initialize(int* argc, char*** argv) {
        // 安装信号处理函数
        install_oneshot_signal_handlers();

        int provided;
        // `MPI_Init_thread()`:初始化MPI执行环境
        // `MPI_THREAD_MULTIPLE`:多个线程可调用MPI
        MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
        // `MPI_Comm_size()`:确定与通信器相关联的组的大小
        // `MPI_COMM_WORLD`:MPI程序中的所有进程
        // 确定总进程数
        MPI_Comm_size(MPI_COMM_WORLD, &partitions_);
        // `MPI_Comm_rank`:确定调用进程在通信器中的序号
        // 确定当前进程id
        MPI_Comm_rank(MPI_COMM_WORLD, &partition_id_);

        if (0 == partition_id_) {
            LOG(INFO) << "thread support level provided by MPI: ";
            switch (provided) {
                case MPI_THREAD_MULTIPLE:
                    LOG(INFO) << "MPI_THREAD_MULTIPLE";
                    break;
                case MPI_THREAD_SERIALIZED:
                    LOG(INFO) << "MPI_THREAD_SERIALIZED";
                    break;
                case MPI_THREAD_FUNNELED:
                    LOG(INFO) << "MPI_THREAD_FUNNELED";
                    break;
                case MPI_THREAD_SINGLE:
                    LOG(INFO) << "MPI_THREAD_SINGLE";
                    break;
                default:
                    CHECK(false) << "unknown mpi thread support level("
                                 << provided << ")";
            }
        }

        if (0 < FLAGS_threads) {  // 若命令行定义了线程数
            threads_ = FLAGS_threads;
        } else if (nullptr !=
                   std::getenv("OMP_NUM_THREADS")) {  // 若环境变量定义了线程数
            threads_ =
                (int)std::strtol(std::getenv("OMP_NUM_THREADS"), nullptr, 10);
        } else {
            // `numa_num_configured_cpus()`:返回系统中已配置的CPU核心数量
            threads_ = numa_num_configured_cpus();
        }
        // `omp_set_dynamic()`:指示即将出现的并行区域中可用的线程数是否可由运行时调整
        // 指明运行时不会动态调整线程数
        omp_set_dynamic(0);
        // `omp_set_num_threads()`:设置即将出现的并行区域中的线程数
        omp_set_num_threads(threads_);
        // `numa_num_configured_nodes()`:获取系统中已配置的NUMA节点数量
        sockets_ = numa_num_configured_nodes();

        // struct bitmask* nodemask = numa_parse_nodestring("all");
        // numa_set_interleave_mask(nodemask);
        // numa_bitmask_free(nodemask);

        if (0 == partition_id_) {
            LOG(INFO) << "threads: " << threads_;
            LOG(INFO) << "sockets: " << sockets_;
            LOG(INFO) << "partitions: " << partitions_;
        }

        initialized = true;
    }

    cluster_info_t(const cluster_info_t&) = delete;
    void operator=(const cluster_info_t&) = delete;

    ~cluster_info_t(void) {
        if (initialized) {
            MPI_Finalize();
        }
    }

   protected:
    cluster_info_t(void)
        : partitions_(-1),
          partition_id_(-1),
          threads_(-1),
          sockets_(-1),
          initialized(false) {}

    /// @brief 初始化标志
    bool initialized;
};

using graph_info_mask_t = uint64_t;

#define GRAPH_INFO_VERTICES (1UL << 0UL)
#define GRAPH_INFO_EDGES (1UL << 1UL)
#define GRAPH_INFO_OUT_DEGREE (1UL << 2UL)

/// @brief 数据图信息
struct graph_info_t {
    // input params

    // 是否是有向图
    bool is_directed_;

    // output params

    /// @brief 总结点数
    vid_t vertices_;
    /// @brief 总边数
    eid_t edges_;
    /// @brief 最大结点ID maximum vertex's id
    vid_t max_v_i_;

    graph_info_t(void)
        : is_directed_(false), vertices_(0), edges_(0), max_v_i_(0) {}

    graph_info_t(bool is_directed)
        : is_directed_(is_directed), vertices_(0), edges_(0), max_v_i_(0) {}
};

// ******************************************************************************* //

// ******************************************************************************* //
// traverse options

/// @brief 遍历模式
enum class traverse_mode_t {
    /// @brief 默认遍历模式 let structure decide
    ORIGIN = 1,
    /// @brief 随机遍历模式
    RANDOM = 2,
    CIRCLE = 3
};

// traverse related
/// @brief 遍历选项
struct traverse_opts_t {
    /// @brief 遍历模式
    traverse_mode_t mode_ = traverse_mode_t::ORIGIN;
    /// @brief 是否自动释放
    bool auto_release_ = false;
};

// ******************************************************************************* //

}  // namespace plato

#endif
