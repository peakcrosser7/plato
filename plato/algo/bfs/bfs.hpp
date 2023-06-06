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

#ifndef __PLATO_ALGO_BFS_HPP__
#define __PLATO_ALGO_BFS_HPP__

#include <cstdint>
#include <cstdlib>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato {
namespace algo {

/// @brief BFS选项
struct bfs_opts_t {
    /// @brief 根结点(起始结点)
    vid_t root_ = 0;
};

/*
 * demo implementation of breadth first search
 *
 * \tparam INCOMING   graph type, with incoming edges
 * \tparam OUTGOING   graph type, with outgoing edges
 *
 * \param in_edges    incoming edges, dcsc, ...
 * \param out_edges   outgoing edges, bcsr, ...
 * \param graph_info  base graph-info
 * \param opts        bfs options
 *
 * \return
 *    visited vertices count
 * */
template <typename INCOMING, typename OUTGOING>
vid_t breadth_first_search(INCOMING& in_edges, OUTGOING& out_edges,
                           const graph_info_t& graph_info,
                           const bfs_opts_t& opts) {
    plato::stop_watch_t watch;
    auto& cluster_info = plato::cluster_info_t::get_instance();

    dualmode_engine_t<INCOMING, OUTGOING> engine(
        std::shared_ptr<INCOMING>(&in_edges, [](INCOMING*) {}),
        std::shared_ptr<OUTGOING>(&out_edges, [](OUTGOING*) {}), graph_info);

    plato::vid_t actives = 1;   // 激活结点数

    // alloc structs used during bfs
    auto visited = engine.alloc_v_subset();     // 已访问结点位图
    auto active_current = engine.alloc_v_subset();  // 当前迭代的激活结点位图
    auto active_next = engine.alloc_v_subset();
    auto parent = engine.template alloc_v_state<plato::vid_t>();    // 结点父结点稠密数据

    // init structs
    plato::vid_t invalid_parent = graph_info.max_v_i_ + 1;  // 无效的父结点ID,用结点最大ID+1表示
    CHECK(invalid_parent != 0) << "vertex id overflow!";
    // 所有结点初始父结点无效
    parent.fill(invalid_parent);
    // 设置源结点的父结点为自身
    parent[opts.root_] = opts.root_;
    // 标记源结点被访问且激活
    visited.set_bit(opts.root_);
    active_current.set_bit(opts.root_);

    // 迭代计算至激活结点数为 0
    for (int epoch_i = 0; 0 != actives; ++epoch_i) {
        using pull_context_t = plato::template mepa_ag_context_t<plato::vid_t>;
        using pull_message_t = plato::template mepa_ag_message_t<plato::vid_t>;
        using push_context_t = plato::template mepa_bc_context_t<plato::vid_t>;
        using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;

        watch.mark("t1");
        active_next.clear();

        // 遍历每条边得到新一轮激活结点数
        actives = engine.template foreach_edges<plato::vid_t, plato::vid_t>(
            // PUSH消息发送函数
            [&](const push_context_t& context, vid_t v_i) {
                context.send(v_i);  // 发送当前结点ID
            },
            // PUSH消息接收函数
            [&](int /*p_i*/, plato::vid_t& msg) {
                plato::vid_t activated = 0;

                auto neighbours = out_edges.neighbours(msg);
                // 遍历接收结点的(出度)邻接表
                for (auto it = neighbours.begin_; neighbours.end_ != it; ++it) {
                    plato::vid_t dst = it->neighbour_;
                    if ((parent[dst] == invalid_parent) &&  // 没有父结点
                        (plato::cas(&parent[dst], invalid_parent, msg))) {  // CAS设置父结点
                        active_next.set_bit(dst);   // 激活当前结点
                        visited.set_bit(dst);   // 设置为已访问
                        ++activated;    // 激活结点数+1
                    }
                }
                return activated;
            },
            // PULL消息发送函数
            [&](const pull_context_t& context, plato::vid_t v_i,
                const adj_unit_list_spec_t& adjs) {
                // 跳过已访问的结点
                if (visited.get_bit(v_i)) {
                    return;
                }
                // 遍历结点的(入度)邻接表
                for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
                    plato::vid_t src = it->neighbour_;
                    if (active_current.get_bit(src)) {  // 源结点此轮激活
                        context.send(pull_message_t{v_i, src}); // 发送当前结点和源结点 
                        break;
                    }
                }
            },
            // PULL消息接收函数
            [&](int, pull_message_t& msg) {
                if (plato::cas(&parent[msg.v_i_], invalid_parent,
                               msg.message_)) { // 设置父结点
                    active_next.set_bit(msg.v_i_);
                    visited.set_bit(msg.v_i_);
                    return 1;
                }
                return 0;
            },
            active_current);

        // 新一轮激活结点的视图
        auto active_view = plato::create_active_v_view(
            out_edges.partitioner()->self_v_view(), active_next);
        // 对新一轮激活结点进行统计
        plato::vid_t __actives =
            active_view.template foreach<plato::vid_t>([&](plato::vid_t v_i) {
                visited.set_bit(v_i);
                return 1;
            });

        CHECK(__actives == actives)
            << "__actives: " << __actives << ", actives: " << actives;
        std::swap(active_current, active_next);

        if (0 == cluster_info.partition_id_) {
            LOG(INFO) << "active_v[" << epoch_i << "] = " << actives
                      << ", cost: " << watch.show("t1") / 1000.0 << "s";
        }
    }

    // BFS访问的结点结果全局同步
    visited.sync();
    // BFS访问的结点个数
    return visited.count();
}

}  // namespace algo
}  // namespace plato

#endif
