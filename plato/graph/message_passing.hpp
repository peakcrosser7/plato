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

#ifndef __PLATO_GRAPH_MESSAGE_PASSING_HPP__
#define __PLATO_GRAPH_MESSAGE_PASSING_HPP__

#include <cstdint>
#include <cstdlib>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "glog/logging.h"
#include "mpi.h"
#include "omp.h"

#include "plato/graph/base.hpp"
#include "plato/parallel/broadcast.hpp"
#include "plato/parallel/bsp.hpp"
#include "plato/util/bitmap.hpp"

namespace plato {

// ******************************************************************************* //
// aggregate message

/// @brief 信息传递聚合的封装消息
/// @tparam MSG 消息类型
template <typename MSG> struct mepa_ag_message_t {
    /// @brief 结点
    vid_t v_i_;
    /// @brief 具体消息
    MSG message_;

    template <typename Ar> void serialize(Ar &ar) { ar &v_i_ &message_; }
};

/// @brief 信息传递聚合发送回调函数
template <typename MSG>
using mepa_ag_send_callback_t =
    std::function<void(const mepa_ag_message_t<MSG> &)>;

/// @brief 信息传递聚合上下文
/// @tparam MSG 消息类型
template <typename MSG> struct mepa_ag_context_t {
    /// @brief 发送回调函数
    mepa_ag_send_callback_t<MSG> send;
};

template <typename MSG, typename ADJ_LIST>
using mepa_ag_merge_t = std::function<void(const mepa_ag_context_t<MSG> &,
                                           vid_t, const ADJ_LIST &)>;

template <typename MSG, typename R>
using mepa_ag_sink_t = std::function<R(int, mepa_ag_message_t<MSG> &)>;

/*
 * Traveling from every vertex, use neighbour's state to generate update
 *message, then send it to target vertex.
 *
 * \tparam MSG        message type
 * \tparam GRAPH      graph structure type
 * \tparam R          return value of sink task
 *
 * \param graph       Graph structure, dcsc, csc, etc...
 * \param merge_task  merge task, run in parallel
 * \param sink_task   message sink task, run in parallel
 * \param opts        bulk synchronous parallel options
 *
 * \return
 *    sum of sink_task return
 **/
template <typename MSG, typename R, typename GRAPH>
R aggregate_message(
    GRAPH &graph,
    mepa_ag_merge_t<MSG, typename GRAPH::adj_unit_list_spec_t> merge_task,
    mepa_ag_sink_t<MSG, R> sink_task, bsp_opts_t bsp_opts = bsp_opts_t()) {

    auto &cluster_info = cluster_info_t::get_instance();
    if (bsp_opts.threads_ <= 0) {
        bsp_opts.threads_ = cluster_info.threads_;
    }

    std::vector<R> reducer_vec(bsp_opts.threads_, R());
    thread_local R *preducer;

    // some compiler is not that smart can inline
    // graph->partitioner()->get_partition_id()
    auto partitioner = graph.partitioner();

    auto bsp_send = [&](bsp_send_callback_t<mepa_ag_message_t<MSG>> send) {
        auto send_callback = [&](const mepa_ag_message_t<MSG> &message) {
            // 将消息发送至对应结点的所在(master)分区
            send(partitioner->get_partition_id(message.v_i_), message);
        };

        mepa_ag_context_t<MSG> context{send_callback};

        // 遍历图中每个结点并根据其邻接表生成消息
        auto traversal = [&](vid_t v_i,
                             const typename GRAPH::adj_unit_list_spec_t &adjs) {
            merge_task(context, v_i, adjs);
            return true;
        };

        size_t chunk_size = 256;
        while (graph.next_chunk(traversal, &chunk_size)) {
        }
    };

    // 接收消息
    auto bsp_recv = [&](int p_i,
                        bsp_recv_pmsg_t<mepa_ag_message_t<MSG>> &pmsg) {
        *preducer += sink_task(p_i, *pmsg);
    };

    traverse_opts_t trvs_opts;
    trvs_opts.mode_ = traverse_mode_t::CIRCLE;
    graph.reset_traversal(trvs_opts);

    // BSP消息传播
    // 图计算对应PULL模式:
    // 发送:每个集群节点遍历本地(mirror)结点及其邻接表计算生成消息,发送消息给master结点所在节点分区
    //      DCSC以边的src划分master结点,因此遍历的本地结点均为mirror结点,而邻接表中的结点均为master结点
    // 接收:master结点接收来自各个mirror结点所在节点分区的消息,进行归约并更新master结点属性,作为下一轮激活结点
    //      下一轮的激活结点就是master结点为源结点的边的终结点,其mirror结点均在此集群节点中
    int rc = fine_grain_bsp<mepa_ag_message_t<MSG>>(
        bsp_send, bsp_recv, bsp_opts,
        // 接收前设置归约值
        [&](void) { preducer = &reducer_vec[omp_get_thread_num()]; });
    CHECK(0 == rc);

    // 多线程归约
    R reducer = R();
    #pragma omp parallel for reduction(+ : reducer)
    for (size_t i = 0; i < reducer_vec.size(); ++i) {
        reducer += reducer_vec[i];
    }

    // 全局归约
    R global_reducer;
    MPI_Allreduce(&reducer, &global_reducer, 1, get_mpi_data_type<R>(), MPI_SUM,
                  MPI_COMM_WORLD);

    return global_reducer;
}

// ******************************************************************************* //

// ******************************************************************************* //
// spread message

/// @brief 消息传递回调函数 (发送节点,消息)->void
/// @tparam MSG 
template <typename MSG>
using mepa_sd_send_callback_t = std::function<void(int, const MSG &)>;

/// @brief 消息传递发送上下文
template <typename MSG> struct mepa_sd_context_t {
    /// @brief 发送函数
    mepa_sd_send_callback_t<MSG> send;
};

template <typename MSG, typename R>
using mepa_sd_sink_t = std::function<R(MSG &)>;

namespace {

/// @brief 绑定的发送任务
/// @tparam F operator()遍历调用函数类型
/// @tparam T 调用参数类型
template <typename F, typename T> 
struct rebind_send_task {
    /// @brief operator()遍历调用函数
    F func;
    /// @brief 调用参数
    T v;

    template <typename... Args>
    inline auto operator()(Args... args) const
        -> decltype(func(v, std::forward<Args>(args)...)) {
        func(v, std::forward<Args>(args)...);
    }
};

/// @brief 绑定发送任务
/// @param func 发送函数
/// @param v 发送的值
/// @return 绑定后的发送任务
template <typename F, typename T>
rebind_send_task<F, T> bind_send_task(F &&func, T &&v) {
    return {std::forward<F>(func), std::forward<T>(v)};
}

} // namespace

/*
 * Traveling from active vertex, generate message, then send it to target 'node'.
 *
 * \tparam MSG        message type
 * \tparam ACTIVE     ACTIVE vertex's container, support bitmap, etc...
 * \tparam R          return value of sink task
 * 
 * \param spread_task spread task, run in parallel, a functor looks like
 *                    void(const mepa_sd_context_t, args...)
 *                    generate message to send
 * \param sink_task   message sink task, run in parallel
 * \param actives     active vertices
 * \param opts        bulk synchronous parallel options
 *
 * \return
 *    sum of sink_task return
 **/
template <typename MSG, typename R, typename ACTIVE, typename SPREAD_FUNC>
R spread_message(ACTIVE &actives, SPREAD_FUNC &&spread_task,
                 mepa_sd_sink_t<MSG, R> sink_task,
                 bsp_opts_t bsp_opts = bsp_opts_t()) {

    auto &cluster_info = cluster_info_t::get_instance();
    if (bsp_opts.threads_ <= 0) {
        bsp_opts.threads_ = cluster_info.threads_;
    }

    // 每个线程的归约器列表
    std::vector<R> reducer_vec(bsp_opts.threads_, R());
    thread_local R *preducer;
    
    /// @brief fine_grain_bsp()函数每次BSP发送执行的函数
    /// @param send BSP执行的回调函数
    auto bsp_send = [&](bsp_send_callback_t<MSG> send) {
        /// @brief actives.next_chunk()中遍历每个元素时调用的函数
        /// @param node 发送到的节点
        /// @param message 发送的消息
        auto send_callback = [&](int node, const MSG &message) {
            send(node, message);
        };

        mepa_sd_context_t<MSG> context{send_callback};

        size_t chunk_size = bsp_opts.local_capacity_;
        auto rebind_traversal =
            bind_send_task(std::forward<SPREAD_FUNC>(spread_task),
                           std::forward<mepa_sd_context_t<MSG>>(context));
        // 对每个缓存分块中的激活结点执行spread_task生成消息
        while (actives.next_chunk(rebind_traversal, &chunk_size)) {
        }
    };

    /// @brief fine_grain_bsp()函数每次BSP接收执行的函数
    /// @param pmsg 接收到的消息
    auto bsp_recv = [&](int /* p_i */, bsp_recv_pmsg_t<MSG> &pmsg) {
        // 对接收的消息进行处理
        *preducer += sink_task(*pmsg);
    };

    // 重置遍历器
    actives.reset_traversal();

    int rc = fine_grain_bsp<MSG>(bsp_send, bsp_recv, bsp_opts, [&](void) {
        // 定义为当前线程位置的归约器元素
        preducer = &reducer_vec[omp_get_thread_num()];
    });
    CHECK(0 == rc);

    R reducer = R();
    // 对每个线程的结果进行归约
    #pragma omp parallel for reduction(+ : reducer)
    for (size_t i = 0; i < reducer_vec.size(); ++i) {
        reducer += reducer_vec[i];
    }

    // MPI进程间归约
    R global_reducer;
    MPI_Allreduce(&reducer, &global_reducer, 1, get_mpi_data_type<R>(), MPI_SUM,
                  MPI_COMM_WORLD);

    return global_reducer;
}

// ******************************************************************************* //

// ******************************************************************************* //
// broadcast message

/// @brief 消息传递广播发送回调函数
template <typename MSG>
using mepa_bc_send_callback_t = std::function<void(const MSG &)>;

/// @brief 消息传递广播上下文
/// @tparam MSG 消息类型
template <typename MSG> struct mepa_bc_context_t {
    /// @brief 发送回调函数
    mepa_bc_send_callback_t<MSG> send;
};

template <typename MSG, typename R>
using mepa_bc_sink_t = std::function<R(int, MSG &)>;

/*
 * Traveling from active vertex, generate message, then broadcast it to every
 *node
 *
 * \tparam MSG          message type
 * \tparam R            return value of sink task
 * \tparam ACTIVE       ACTIVE vertex's container, support bitmap, etc...
 * \tparam SPREAD_FUNC  spread task type
 *
 * \param spread_task spread task, run in parallel, a functor looks like
 *                    void(const mepa_bc_context_t, args...)
 *                    generate message to broadcast
 * \param sink_task   message sink task, run in parallel
 * \param actives     active vertices
 * \param opts        bulk synchronous parallel options
 *
 * \return
 *    sum of sink_task return
 **/
template <typename MSG, typename R, typename ACTIVE, typename SPREAD_FUNC>
R broadcast_message(ACTIVE &actives, SPREAD_FUNC &&spread_task,
                    mepa_bc_sink_t<MSG, R> sink_task,
                    bc_opts_t bc_opts = bc_opts_t()) {

    auto &cluster_info = cluster_info_t::get_instance();
    if (bc_opts.threads_ <= 0) {
        bc_opts.threads_ = cluster_info.threads_;
    }

    std::vector<R> reducer_vec(bc_opts.threads_, R());
    thread_local R *preducer;

    // 每个激活master结点发送消息的函数
    auto __send = [&](bc_send_callback_t<MSG> send) {
        auto send_callback = [&](const MSG &message) { send(message); };

        mepa_bc_context_t<MSG> context{send_callback};

        size_t chunk_size = bc_opts.local_capacity_;
        auto rebind_traversal =
            bind_send_task(std::forward<SPREAD_FUNC>(spread_task),
                           std::forward<mepa_bc_context_t<MSG>>(context));
        while (actives.next_chunk(rebind_traversal, &chunk_size)) {
        }
    };

    // 接收消息函数
    auto __recv = [&](int p_i, bc_recv_pmsg_t<MSG> &pmsg) {
        *preducer += sink_task(p_i, *pmsg);
    };

    actives.reset_traversal();

    // 消息广播
    // 图计算对应PUSH模式:
    // 发送:每个集群节点遍历本地激活的master结点生成消息,发送消息给其mirror结点
    //      BCSR以边的dst划分master结点,因此mirror结点可能在其他所有集群节点,均要发送,即以环形顺序广播消息
    // 接收:每个集群节点的mirror结点接收来自master结点所在节点分区的消息,遍历其邻接表,更新邻结点属性并作为下一轮结点激活
    //      下一轮的激活结点是以mirror结点为源结点的边的终结点,其master结点均在此集群节点中
    int rc = broadcast<MSG>(__send, __recv, bc_opts, [&](void) {
        preducer = &reducer_vec[omp_get_thread_num()];
    });
    CHECK(0 == rc);

    R reducer = R();
    // 多线程归约
    #pragma omp parallel for reduction(+ : reducer)
    for (size_t i = 0; i < reducer_vec.size(); ++i) {
        reducer += reducer_vec[i];
    }

    // 全局归约
    R global_reducer;
    MPI_Allreduce(&reducer, &global_reducer, 1, get_mpi_data_type<R>(), MPI_SUM,
                  MPI_COMM_WORLD);

    return global_reducer;
}

// ******************************************************************************* //

} // namespace plato

#endif
