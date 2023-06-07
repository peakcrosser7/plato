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

#ifndef __PLATO_GRAPH_PARTITION_SEQUENCE_HPP__
#define __PLATO_GRAPH_PARTITION_SEQUENCE_HPP__

#include <cstdint>
#include <cstdlib>

#include <atomic>
#include <functional>
#include <memory>
#include <type_traits>

#include "glog/logging.h"
#include "mpi.h"

#include "plato/graph/base.hpp"
#include "plato/parallel/mpi.hpp"

namespace plato {

namespace { // helper function

/// @brief 初始化集群节点的结点偏移数组
/// @param[out] poffset 机器节点对应的结点偏移量数组
/// @param degrees 结点度数数组
/// @param vertices 结点总数
/// @param edges 边总数
/// @param alpha 结点的计算权重
template <typename DT>
void __init_offset(std::vector<vid_t> *poffset, const DT *degrees,
                   vid_t vertices, eid_t edges, int alpha) {
    auto &cluster_info = cluster_info_t::get_instance();
    // 剩余待分配的数量
    uint64_t remained_amount = edges + vertices * (uint64_t)alpha;
    // 每个节点期望分配的数量
    uint64_t expected_amount = 0;

    if (0 == cluster_info.partition_id_) {
        LOG(INFO) << "total_amount: " << remained_amount
                  << ", alpha: " << alpha;
    }

    poffset->clear();
    poffset->resize(cluster_info.partitions_ + 1, 0);
    for (int p_i = 0; p_i < cluster_info.partitions_; ++p_i) {
        // 每个节点的期望数量均为当前剩余数量在节点平分的数量
        expected_amount = remained_amount / (cluster_info.partitions_ - p_i);

        uint64_t amount = 0;    // 每个节点实际分配数量
        for (vid_t v_i = poffset->at(p_i); v_i < vertices; ++v_i) {
            // 加上当前结点的数量
            amount += (alpha + degrees[v_i]);
            if (amount >= expected_amount) {    // 数量达到预期数量
                // 结点ID要页面对齐
                poffset->at(p_i + 1) = v_i / PAGESIZE * PAGESIZE;
                break;
            }
        }
        if ((cluster_info.partitions_ - 1) == p_i) {
            poffset->at(cluster_info.partitions_) = vertices;
        }

        remained_amount -= amount;
        if (0 == cluster_info.partition_id_) {
            LOG(INFO) << "partition-" << p_i << ": [" << poffset->at(p_i) << ","
                      << poffset->at(p_i + 1) << ")"
                      << ", amount: " << amount;
        }
    }
}

/// @brief 检查结点偏移量数组的全局一致性
/// @param offset_ 结点偏移量数组
void __check_consistency(const std::vector<vid_t> &offset_) {
    std::vector<vid_t> offset(offset_.size());

    MPI_Allreduce(offset_.data(), offset.data(), offset_.size(),
                  get_mpi_data_type<vid_t>(), MPI_MAX, MPI_COMM_WORLD);

    for (size_t i = 0; i < offset.size(); ++i) {
        CHECK(offset[i] == offset_[i]);
    }
}

} // namespace

/// @brief 序列划分视图 sequence partition view
class sequence_v_view {
  public:
    // *******************************************************************************
    // // required types & methods

    // traverse related
    void reset_traversal(const traverse_opts_t &opts = traverse_opts_t());

    /*
     * process a chunk of vertices, thread-safe
     *
     * \tparam TRAVERSAL  traversal functor, it should implement the method:
     *                    <tt>void operator() (vid_t)<\tt>;
     *
     * \return true - traverse at lease one edge, false - no more edges to
     *traverse.
     **/
    template <typename TRAVERSAL>
    bool next_chunk(TRAVERSAL &&traversal, size_t *chunk_size);

    // *******************************************************************************
    // //

    sequence_v_view(const sequence_v_view &) = delete;
    sequence_v_view &operator=(const sequence_v_view &) = delete;

    sequence_v_view(vid_t start, vid_t end);
    sequence_v_view(sequence_v_view &&other);

    vid_t start() { return start_; }
    vid_t end() { return end_; }

  protected:
    /// @brief 起始结点ID
    vid_t start_;
    /// @brief 终止结点ID
    vid_t end_;
    /// @brief 遍历索引
    std::atomic<vid_t> traverse_i_;
};

/*
 * Partitioner try to keep each partitions' computation work balanced
 * vertexId must be compacted
 *
 * references:
 *  Julian Shun, Guy E Blelloch. Ligra: A Lightweight Graph Processing
 *  Framework for Shared Memory
 *
 *  Xiaowei Zhu, Wenguang Chen, etc. Gemini: A Computation-Centric Distributed
 *  Graph Processing System
 **/

// edge belong to source node's partition
/// @brief 根据边的源结点的划分序列
class sequence_balanced_by_source_t {
  public:
    // *******************************************************************************
    // // required types & methods

    // get edge's partition
    /// @brief  根据源结点获取边所在的子图分区(集群节点) get edge's partition
    /// @param src 源结点
    /// @param dst 终结点
    /// @return 所在的子图分区(集群节点)
    inline int get_partition_id(vid_t src, vid_t /*dst*/) {
        return get_partition_id(src);
    }

    // get vertex's partition
    /// @brief 获取结点的子图分区(集群节点) get vertex's partition
    /// @param v_i 结点ID
    /// @return 所在的子图分区(集群节点)
    inline int get_partition_id(vid_t v_i) {
        for (size_t p_i = 0; p_i < (offset_.size() - 1); ++p_i) {
            if (v_i >= offset_[p_i] && v_i < offset_[p_i + 1]) {
                return p_i;
            }
        }
        CHECK(false) << "can not find which partition " << v_i << " belong";
        // add abort() to make gcc 6 happy. Otherwise compile failed due to
        // -Werror=return-type.
        abort();
    }

    /// @brief 获取本地节点对应的结点分区视图
    sequence_v_view self_v_view(void) {
        auto &cluster_info = cluster_info_t::get_instance();
        return sequence_v_view(offset_[cluster_info.partition_id_],
                               offset_[cluster_info.partition_id_ + 1]);
    }

    // *******************************************************************************
    // //

    /*
     * constructor
     *
     * \param degrees   each vertex's degrees
     * \param vertices  vertex number of the graph
     * \param vertices  edge number of the graph
     * \param alpha     vertex's weight of computation, default: -1, means
     *                  alpha = 8 * (partitions - 1)
     **/
    template <typename DT>
    sequence_balanced_by_source_t(const DT *degrees, vid_t vertices,
                                  eid_t edges, int alpha = -1) {
        if (-1 == alpha) {
            auto &cluster_info = cluster_info_t::get_instance();
            alpha = 8 * (cluster_info.partitions_ - 1);
        }
        __init_offset(&offset_, degrees, vertices, edges, alpha);
    }

    sequence_balanced_by_source_t(const std::vector<vid_t> &offset)
        : offset_(offset) {}

    void check_consistency(void) { __check_consistency(offset_); }

    // *******************************************************************************
    // //

    /// @brief 集群节点对应的结点偏移量数组
    std::vector<vid_t> offset_;

    // *******************************************************************************
    // //
};

// edge belong to destination node's partition
/// @brief 根据边的终结点的划分序列
class sequence_balanced_by_destination_t {
  public:
    // *******************************************************************************
    // // required types & methods

    /// @brief  根据终结点获取边所在的子图分区(集群节点) get edge's partition
    /// @param src 源结点
    /// @param dst 终结点
    /// @return 所在的子图分区(集群节点)
    inline int get_partition_id(vid_t /*src*/, vid_t dst) {
        return get_partition_id(dst);
    }

    /// @brief 获取结点的子图分区(集群节点) get vertex's partition
    /// @param v_i 结点ID
    /// @return 所在的子图分区(集群节点)
    inline int get_partition_id(vid_t v_i) {
        for (size_t p_i = 0; p_i < (offset_.size() - 1); ++p_i) {
            if (v_i >= offset_[p_i] && v_i < offset_[p_i + 1]) {
                return p_i;
            }
        }
        CHECK(false) << "can not find which partition " << v_i << " belong";
        // add abort() to make gcc 6 happy. Otherwise compile failed due to
        // -Werror=return-type.
        abort();
    }

    /// @brief 返回本地节点的结点分区视图
    sequence_v_view self_v_view(void) {
        auto &cluster_info = cluster_info_t::get_instance();
        return sequence_v_view(offset_[cluster_info.partition_id_],
                               offset_[cluster_info.partition_id_ + 1]);
    }

    // *******************************************************************************
    // //

    /*
     * constructor
     *
     * \param degrees   each vertex's degrees
     * \param vertices  vertex number of the graph
     * \param edges     edge number of the graph
     * \param alpha     vertex's weight of computation, default: -1, means
     *                  alpha = 8 * (partitions - 1)
     **/
    template <typename DT>
    sequence_balanced_by_destination_t(const DT *degrees, vid_t vertices,
                                       eid_t edges, int alpha = -1) {
        if (-1 == alpha) {
            auto &cluster_info = cluster_info_t::get_instance();
            alpha = 8 * (cluster_info.partitions_ - 1);
        }
        __init_offset(&offset_, degrees, vertices, edges, alpha);
    }

    sequence_balanced_by_destination_t(const std::vector<vid_t> &offset)
        : offset_(offset) {}

    /// @brief 检查结点偏移量数组offset_的集群全局一致性
    void check_consistency(void) { __check_consistency(offset_); }

    // *******************************************************************************
    // //

    /// @brief 集群节点对应的结点偏移量数组
    std::vector<vid_t> offset_;

    // *******************************************************************************
    // //
};

/// @brief 判断子图分区类型是否是序列类型(有记录集群节点对应的结点范围的数组)
/// @tparam PART 子图分区类型
template <typename PART> constexpr bool is_seq_part(void) {
    return std::is_same<PART, sequence_balanced_by_source_t>::value ||
           std::is_same<PART, sequence_balanced_by_destination_t>::value;
}

// ************************************************************************************
// // implementations

sequence_v_view::sequence_v_view(vid_t start, vid_t end)
    : start_(start), end_(end), traverse_i_(start_) {}

sequence_v_view::sequence_v_view(sequence_v_view &&other)
    : start_(other.start_), end_(other.end_), traverse_i_(start_) {}

/// @brief 重置遍历
/// @param opts 遍历选项
void sequence_v_view::reset_traversal(const traverse_opts_t &opts) {
    CHECK(opts.mode_ == traverse_mode_t::ORIGIN);
    traverse_i_.store(start_, std::memory_order_relaxed);
}

/// @brief 处理一个分块的结点(线程安全)
/// @param traversal 遍历操作函数
/// @param[in,out] chunk_size 分块大小 
/// @return 至少有结点被遍历
template <typename TRAVERSAL>
bool sequence_v_view::next_chunk(TRAVERSAL &&traversal, size_t *chunk_size) {
    vid_t range_start =
        traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);
    ;

    if (range_start >= end_) {
        return false;
    }
    if (range_start + *chunk_size > end_) {
        *chunk_size = end_ - range_start;
    }

    vid_t range_end = range_start + *chunk_size;
    // 遍历范围区间里的每个结点并执行操作
    for (vid_t range_i = range_start; range_i < range_end; ++range_i) {
        traversal(range_i);
    }
    return true;
}

// ************************************************************************************
// //

} // namespace plato

#endif
