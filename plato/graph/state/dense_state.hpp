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

#ifndef __PLATO_GRAPH_DENSE_STATE_HPP__
#define __PLATO_GRAPH_DENSE_STATE_HPP__

#include <cstdint>
#include <memory>
#include <atomic>
#include <random>
#include <type_traits>

#include "omp.h"
#include "mpi.h"
#include "glog/logging.h"

#include "plato/graph/base.hpp"
#include "plato/graph/state/detail.hpp"
#include "plato/graph/partition/sequence.hpp"
#include "plato/util/bitmap.hpp"
#include "plato/util/mmap_alloc.hpp"

namespace plato {

/// @brief 稠密状态数据
/// @tparam PART_IMPL 结点划分序列类型
/// @tparam BITMAP 位图类型
template <typename T, typename PART_IMPL, typename ALLOC = mmap_allocator_t<T>,
          typename BITMAP = bitmap_t<>>
class dense_state_t {
   protected:
    using traits_ =
        typename std::allocator_traits<ALLOC>::template rebind_traits<T>;

   public:
    // static_assert(std::is_trivial<T>::value &&
    // std::is_standard_layout<T>::value,
    //     "dense_state_t only support pod-type");

    // *******************************************************************************
    // // required types & methods

    using value_t = T;
    using partition_t = PART_IMPL;
    using allocator_type = typename traits_::allocator_type;
    using bitmap_spec_t = BITMAP;

    /*
     * call 'func' on each vertices belong to this partition
     *
     * \param func    user provide vertex process logic
     * \param active  active vertices set
     *
     * \return
     *    sum of every func's return value
     **/
    // template <typename R>
    // R foreach(std::function<R(vid_t, T*)> func, bitmap_t active);

    // fill all vertices value to T belong to this partition
    void fill(const T& value);

    /*
     * returns a reference to the element at position n in the state container
     **/
    T& operator[](size_t n);
    const T& operator[](size_t n) const;

    // get partitioner
    std::shared_ptr<partition_t> partitioner(void) { return partitioner_; }

    // traverse related

    // return true -- continue travel, false -- stop.
    using traversal_t = std::function<bool(vid_t, T*)>;

    // start travel from all/subset of the vertices
    void reset_traversal(std::shared_ptr<bitmap_spec_t> pactive = nullptr,
                         const traverse_opts_t& opts = traverse_opts_t());

    /*
     * process a chunk of vertices
     *
     * \param traversal   callback function process on each vertex's state
     * \param chunk_size  process at most chunk_size vertices
     *
     * \return
     *    true  -- process at lease one vertex
     *    false -- no more vertex to process
     **/
    bool next_chunk(traversal_t traversal, size_t* chunk_size);

    /*
     * process active vertices in parallel
     *
     * \param process     user define callback for each eligible vertex
     *                    R(vid_t v_i, value_t* value)
     * \param pactives    bitmap used for filter subset of the vertex
     * \param chunk_size  at most process 'chunk_size' chunk at a batch
     *
     * \return
     *        sum of 'process' return
     **/
    template <typename R, typename PROCESS>
    R foreach (PROCESS&& process, bitmap_spec_t* pactives = nullptr,
               size_t chunk_size = PAGESIZE);

    // ******************************************************************************* //

    /*
     * constructor
     *
     * \param max_v_i       maximum vertex's id
     * \param partitioner
     * \param alloc         allocator for internal storage
     **/
    dense_state_t(vid_t max_v_i, std::shared_ptr<partition_t> partitioner,
                  const allocator_type& alloc = ALLOC());

    dense_state_t(dense_state_t&& other);
    dense_state_t& operator=(dense_state_t&& other);

    dense_state_t(const dense_state_t&) = delete;
    dense_state_t& operator=(const dense_state_t&) = delete;

    ~dense_state_t(void);

    // std::shared_ptr<T> data(void) { return data_; }
    allocator_type& allocator(void) { return allocator_; }

   protected:
    /// @brief 最大结点ID
    vid_t max_v_i_;
    allocator_type allocator_;

    /// @brief 结点数据数组
    value_t* data_;
    /// @brief 结点划分序列
    std::shared_ptr<partition_t> partitioner_;

    vid_t traverse_start_;
    vid_t traverse_end_;
    /// @brief 遍历索引
    std::atomic<vid_t> traverse_i_;
    /// @brief 遍历的激活结点位图
    std::shared_ptr<bitmap_spec_t> traverse_active_;
    /// @brief (用于随机遍历的)遍历范围数组
    std::vector<std::pair<vid_t, vid_t>> traverse_range_;
    /// @brief 遍历选项
    traverse_opts_t traverse_opts_;

    /// @brief 集群信息
    const cluster_info_t& cluster_info_;

    // SFINAE only works for deduced template arguments, it's tricky here

    /// @brief 子图分区的起点(序列分区实现)
    /// @param part 子图分区序列
    /// @return 当前集群节点对应子图的起始结点ID
    template <typename PART>
    typename std::enable_if<is_seq_part<PART>(), vid_t>::type part_start(
        std::shared_ptr<PART> part) {
        return part->offset_[cluster_info_.partition_id_];
    }

    /// @brief 子图分区的终点(序列分区实现)
    /// @param part 子图分区序列
    /// @return 当前集群节点对应子图的终止结点ID
    template <typename PART>
    typename std::enable_if<is_seq_part<PART>(), vid_t>::type part_end(
        std::shared_ptr<PART> part) {
        return part->offset_[cluster_info_.partition_id_ + 1];
    }

    /// @brief 子图分区的起点(非序列分区实现)
    /// @return 0
    template <typename PART>
    typename std::enable_if<!is_seq_part<PART>(), vid_t>::type part_start(
        std::shared_ptr<PART>) {
        return 0;
    }
    /// @brief 子图分区的终点(非序列分区实现报错)
    /// @return 全局结点最大ID
    template <typename PART>
    typename std::enable_if<!is_seq_part<PART>(), vid_t>::type part_end(
        std::shared_ptr<PART>) {
        return max_v_i_;
    }
};

// ************************************************************************************ //
// implementations

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::dense_state_t(
    vid_t max_v_i, std::shared_ptr<partition_t> partitioner,
    const allocator_type& alloc)
    : max_v_i_(max_v_i + 1),
      allocator_(alloc),
      partitioner_(partitioner),
      traverse_start_(0),
      traverse_end_(0),
      traverse_i_(0),
      cluster_info_(cluster_info_t::get_instance()) {
    data_ = allocator_.allocate(max_v_i_);
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::dense_state_t(dense_state_t&& ot)
    : max_v_i_(ot.max_v_i_),
      allocator_(ot.allocator_),
      partitioner_(std::move(ot.partitioner_)),
      traverse_start_(ot.traverse_start_),
      traverse_end_(ot.traverse_end_),
      traverse_i_(ot.traverse_i_.load()),
      traverse_active_(std::move(ot.traverse_active_)),
      cluster_info_(ot.cluster_info_) {
    data_ = ot.data_;
    ot.data_ = nullptr;
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>&
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::operator=(dense_state_t&& ot) {
    if (nullptr != data_) {
        allocator_.deallocate(data_, max_v_i_);
    }

    max_v_i_ = ot.max_v_i_;
    allocator_ = ot.allocator_;
    partitioner_ = std::move(ot.partitioner_);
    traverse_start_ = ot.traverse_start_;
    traverse_end_ = ot.traverse_end_;
    traverse_i_.store(ot.traverse_i_);
    traverse_active_ = std::move(ot.traverse_active_);

    data_ = ot.data_;
    ot.data_ = nullptr;

    // cluster_info_ ??
    return *this;
}

template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::~dense_state_t(void) {
    if (nullptr != data_) {
        allocator_.deallocate(data_, max_v_i_);
    }
}

/// @brief 返回第n个结点的数据
/// @param n 结点ID
template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
inline T& dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::operator[](size_t n) {
    // CHECK(n < max_v_i_);
    return data_[n];
}

/// @brief 返回第n个结点的数据
/// @param n 结点ID
template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
inline const T& dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::operator[](
    size_t n) const {
    // CHECK(n < max_v_i_);
    return data_[n];
}

/// @brief 对所有结点的数据赋值
/// fill all vertices value to T belong to this partition 
template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
void dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::fill(const T& value) {
    auto traversal = [&](vid_t v_i, T* pval) {
        *pval = value;
        return true;
    };

    reset_traversal();
#pragma omp parallel
    {
        size_t chunk_size = 64;
        while (next_chunk(traversal, &chunk_size)) {
        }
    }
}

/// @brief 重置遍历 start travel from all/subset of the vertices
/// @param pactive 
/// @param opts 遍历选项
template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
void dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::reset_traversal(
    std::shared_ptr<bitmap_spec_t> pactive, const traverse_opts_t& opts) {
    if (false == is_seq_part<partition_t>()) {
        LOG(WARNING) << "scan all vertices to perform foreach op. sequence "
                        "partition/sparse state is a better option";
    }

    // 本地节点的子图起止结点
    traverse_start_ = part_start(partitioner_);
    traverse_end_ = part_end(partitioner_);
    traverse_active_ = pactive;

    traverse_i_.store(traverse_start_, std::memory_order_relaxed);

    // prebuild chunk blocks for random traverse
    if (is_seq_part<partition_t>() && traverse_mode_t::RANDOM == opts.mode_) {
        if ((traverse_opts_.mode_ == opts.mode_) && traverse_range_.size()) {
            return;  // use cached range
        }
        traverse_opts_ = opts;

        {
            // 子图包含的结点数
            vid_t vertices = traverse_end_ - traverse_start_;
            // 按照CHUNKSIZE将结点划分到桶中
            size_t buckets = (size_t)vertices / CHUNKSIZE +
                             std::min(vertices % CHUNKSIZE, (vid_t)1);

            traverse_range_.resize(buckets);
            vid_t v_start = traverse_start_;

            for (size_t i = 0; i < buckets; ++i) {
                if ((buckets - 1) == i) {
                    traverse_range_[i].first = v_start;
                    traverse_range_[i].second = traverse_end_;
                } else {
                    traverse_range_[i].first = v_start;
                    traverse_range_[i].second = v_start + CHUNKSIZE;
                }
                v_start = traverse_range_[i].second;
            }
        }

        traverse_i_.store(0, std::memory_order_relaxed);
        // 打乱顺序
        std::random_shuffle(traverse_range_.begin(), traverse_range_.end());
    }
}

/// @brief 处理一个分块的(激活)结点并执行操作(线程安全)
/// @param traversal 遍历操作函数
/// @param[in,out] chunk_size 分块大小
/// @return 是否有结点被遍历
template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
bool dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::next_chunk(
    traversal_t traversal, size_t* chunk_size) {
    // 对序列分区的随机遍历(每次遍历chunk_size个范围区间)
    if (is_seq_part<partition_t>() &&
        traverse_mode_t::RANDOM == traverse_opts_.mode_) {
        vid_t range_start =
            traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);

        if (range_start >= traverse_range_.size()) {
            return false;
        }
        if (range_start + *chunk_size > traverse_range_.size()) {
            *chunk_size = traverse_range_.size() - range_start;
        }

        vid_t range_end = range_start + *chunk_size;
        // 遍历对应的范围区间
        for (vid_t range_i = range_start; range_i < range_end; ++range_i) {
            vid_t v_start = traverse_range_[range_i].first;
            vid_t v_end = traverse_range_[range_i].second;
            // 遍历区间中的所有结点
            for (vid_t v_i = v_start; v_i < v_end; ++v_i) {
                // 要求结点激活
                if (nullptr == traverse_active_ ||
                    traverse_active_->get_bit(v_i)) {
                    // 操作结点及其数据
                    if (false == traversal(v_i, &data_[v_i])) {
                        return true;
                    }
                }
            }
        }
    } else {    // 其他遍历模式(遍历chunk_size个结点)
        vid_t v_start =
            traverse_i_.fetch_add(*chunk_size, std::memory_order_relaxed);

        if (v_start >= traverse_end_) {
            return false;
        }
        if (v_start + *chunk_size > traverse_end_) {
            *chunk_size = traverse_end_ - v_start;
        }

        vid_t v_end = v_start + *chunk_size;

        if (nullptr == traverse_active_) {  // scan all vertices
            if (is_seq_part<partition_t>()) {
                for (vid_t v_i = v_start; v_i < v_end; ++v_i) {
                    if (false == traversal(v_i, &data_[v_i])) {
                        return true;
                    }
                }
            } else {  // check vertex in this partition or not
                for (vid_t v_i = v_start; v_i < v_end; ++v_i) {
                    // 要求该结点位于本地节点
                    if (cluster_info_.partition_id_ !=
                        partitioner_->get_partition_id(v_i)) {
                        continue;
                    }
                    if (false == traversal(v_i, &data_[v_i])) {
                        return true;
                    }
                }
            }
        } else {    // 有激活结点位图
            const size_t chunk = 64;

            if (is_seq_part<partition_t>()) {   // 分区序列
                vid_t v_i = v_start;

                {  // first non-padding chunk
                    vid_t __v_end = (v_start + chunk) / chunk * chunk;

                    if (__v_end > v_end) {
                        __v_end = v_end;
                    }

                    for (; v_i < __v_end; ++v_i) {
                        // 只遍历激活结点
                        if (0 == traverse_active_->get_bit(v_i)) {
                            continue;
                        }
                        if (false == traversal(v_i, &data_[v_i])) {
                            return true;
                        }
                    }
                }

                for (; v_i < v_end; v_i += chunk) {
                    uint64_t word = traverse_active_->data_[word_offset(v_i)];
                    vid_t __v_i = v_i;

                    while (word) {
                        if (word & 0x01) {
                            if (false == traversal(__v_i, &data_[__v_i])) {
                                return true;
                            }
                        }

                        ++__v_i;
                        if (__v_i >= v_end) {
                            break;
                        }
                        word = word >> 1;
                    }
                }
            } else {    // 非分区序列
                vid_t v_i = v_start;

                {  // first non-padding chunk
                    vid_t __v_end = (v_start + chunk) / chunk * chunk;

                    if (__v_end > v_end) {
                        __v_end = v_end;
                    }

                    for (; v_i < __v_end; ++v_i) {
                        // 判断结点位于本地节点且被激活才遍历
                        if ((cluster_info_.partition_id_ !=
                             partitioner_->get_partition_id(v_i)) ||
                            (0 == traverse_active_->get_bit(v_i))) {
                            continue;
                        }
                        if (false == traversal(v_i, &data_[v_i])) {
                            return true;
                        }
                    }
                }

                for (; v_i < v_end; v_i += chunk) {
                    uint64_t word = traverse_active_->data_[word_offset(v_i)];
                    vid_t __v_i = v_i;

                    while (word) {
                        if ((cluster_info_.partition_id_ ==
                             partitioner_->get_partition_id(__v_i)) &&
                            (word & 0x01)) {
                            if (false == traversal(__v_i, &data_[__v_i])) {
                                return true;
                            }
                        }

                        ++__v_i;
                        if (__v_i >= v_end) {
                            break;
                        }
                        word = word >> 1;
                    }
                }
            }
        }
    }

    return true;
}

/// @brief 遍历每个激活结点的数据
/// @param process 处理函数
/// @param pactives 激活结点位图
/// @param chunk_size 分块大小
/// @return 处理函数的归约值
template <typename T, typename PART_IMPL, typename ALLOC, typename BITMAP>
template <typename R, typename PROCESS>
R dense_state_t<T, PART_IMPL, ALLOC, BITMAP>::foreach (PROCESS&& process,
                                                       bitmap_spec_t * pactives,
                                                       size_t chunk_size) {
    return __foreach<R>(this, std::forward<PROCESS>(process), pactives,
                        chunk_size);
}

// ************************************************************************************ //

}  // namespace plato

#endif
