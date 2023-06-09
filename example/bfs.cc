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

#include <cstdint>
#include <cstdlib>
#include <utility>

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"

DEFINE_string(input,       "",     "input file, in csv format, without edge data");
DEFINE_bool(is_directed,   false,  "is graph directed or not");
DEFINE_uint32(root,        0,      "start bfs from which vertex");
DEFINE_int32(alpha,        -1,     "alpha value used in sequence balance partition");
DEFINE_bool(part_by_in,    false,  "partition by in-degree");
DEFINE_uint32(type,        0,      "0 -- always pull, 1 -- push-pull, else -- push");

bool string_not_empty(const char *, const std::string &value) {
    if (0 == value.length()) {
        return false;
    }
    return true;
}

void init(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
}

int main(int argc, char **argv) {
    using bcsr_spec_t          = plato::bcsr_t<plato::empty_t, plato::sequence_balanced_by_destination_t>;
    using dcsc_spec_t          = plato::dcsc_t<plato::empty_t, plato::sequence_balanced_by_source_t>;
    using partition_bcsr_t     = bcsr_spec_t::partition_t;
    using state_parent_t       = plato::dense_state_t<plato::vid_t, partition_bcsr_t>;
    using bitmap_spec_t        = plato::bitmap_t<>;

    plato::stop_watch_t watch;
    auto &cluster_info = plato::cluster_info_t::get_instance();

    // 初始化
    init(argc, argv);
    cluster_info.initialize(&argc, &argv);

    watch.mark("t0");

    plato::graph_info_t graph_info(FLAGS_is_directed);
    // 根据文件路径构建双模式(BCSR和DCSC)的图结构
    auto graph = plato::create_dualmode_seq_from_path<plato::empty_t>(
        &graph_info, FLAGS_input, plato::edge_format_t::CSV,
        plato::dummy_decoder<plato::empty_t>, FLAGS_alpha, FLAGS_part_by_in);

    watch.mark("t1");
    // 生成稠密的结点出度数据
    auto out_degrees = plato::generate_dense_out_degrees_fg<plato::vid_t>(
        graph_info, graph.first, true);
    if (0 == cluster_info.partition_id_) {
        LOG(INFO) << "generate out-degrees from graph cost: "
                  << watch.show("t1") / 1000.0 << "s";
    }

    plato::eid_t edges = graph_info.edges_;
    if (false == graph_info.is_directed_) {
        edges = edges * 2;
    }

    plato::vid_t actives = 1;
    // 父结点稠密数据
    state_parent_t parent(graph_info.max_v_i_, graph.first.partitioner());
    // 已访问结点位图
    bitmap_spec_t visited(graph_info.vertices_);
    // 此轮迭代激活结点位图
    std::shared_ptr<bitmap_spec_t> active_current(
        new bitmap_spec_t(graph_info.vertices_));
    // 下一轮迭代激活结点位图
    std::shared_ptr<bitmap_spec_t> active_next(
        new bitmap_spec_t(graph_info.vertices_));

    // BFS初始化
    visited.set_bit(FLAGS_root);
    active_current->set_bit(FLAGS_root);
    parent.fill(graph_info.vertices_);
    parent[FLAGS_root] = FLAGS_root;

    // 本地节点的master结点分区视图
    auto partition_view = graph.first.partitioner()->self_v_view();

    watch.mark("t1");
    watch.mark("t2");

    bool is_sparse = true;  // 激活边是否稀疏
    plato::eid_t active_edges = 0;  // 此轮迭代激活结点数
    plato::vid_t last_actives = 0;  // 上一轮迭代激活结点数
    // 迭代至激活结点数为0
    for (int epoch_i = 0; 0 != actives; ++epoch_i) {
        if (0 == cluster_info.partition_id_) {
            LOG(INFO) << "active_v[" << epoch_i << "] = " << last_actives
                      << ", active_e[" << active_edges << "/" << is_sparse
                      << "], cost: " << watch.show("t1") / 1000.0 << "s";
            ;
            last_actives = actives;
        }

        watch.mark("t1");
        // 本地节点的激活的master结点视图
        auto active_view =
            plato::create_active_v_view(partition_view, *active_current);

        { // count active edges 统计全局激活边数量
            active_edges = 0;
            if (1 == FLAGS_type) {  // PULL
                out_degrees.reset_traversal(active_current);
                #pragma omp parallel reduction(+ : active_edges)
                {
                    size_t chunk_size = 4 * PAGESIZE;
                    plato::eid_t __active_edges = 0;

                    while (out_degrees.next_chunk(
                        [&](plato::vid_t v_i, plato::vid_t *degrees) {
                            __active_edges += (*degrees);
                            return true;
                        },
                        &chunk_size)) {
                    }
                    active_edges += __active_edges;
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, &active_edges, 1,
                          plato::get_mpi_data_type<plato::eid_t>(), MPI_SUM,
                          MPI_COMM_WORLD);

            // switch between aggregate_message and spread_message
            is_sparse = (active_edges < edges / 20);    // 根据激活边比例确定稀疏情况
        }

        active_next->clear();
        if (0 == FLAGS_type || false == is_sparse) { // pull
            using context_spec_t = plato::mepa_ag_context_t<plato::vid_t>;
            using message_spec_t = plato::mepa_ag_message_t<plato::vid_t>;

            watch.mark("t11");
            visited.sync();

            if (0 == cluster_info.partition_id_) {
                LOG(INFO) << "bitmap sync time: " << watch.show("t11") / 1000.0
                          << "s";
            }

            plato::bsp_opts_t opts;
            opts.local_capacity_ = 32 * PAGESIZE;
            // 聚合计算得到新一轮激活结点数
            actives = plato::aggregate_message<plato::vid_t, int, dcsc_spec_t>(
                // 入边方向图结构
                graph.second,
                // PULL消息发送函数
                [&](const context_spec_t &context, plato::vid_t v_i,
                    const adj_unit_list_spec_t &adjs) {
                    // 跳过已访问的结点
                    if (visited.get_bit(v_i)) {
                        return;
                    }
                    // 遍历结点的(入度)邻接表
                    for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
                        plato::vid_t src = it->neighbour_;
                        // 源结点此轮激活
                        if (active_current->get_bit(src)) {
                            // 发送当前结点和源结点到其master节点
                            context.send(message_spec_t{v_i, src});
                            break;
                        }
                    }
                },
                // PULL消息接收函数
                [&](int /*p_i*/, message_spec_t &message) {
                    // CAS设置接收结点的父结点(更新master结点状态)
                    if (plato::cas(&parent[message.v_i_], graph_info.vertices_,
                                   message.message_)) {
                        active_next->set_bit(message.v_i_);
                        visited.set_bit(message.v_i_);
                        return 1;
                    }
                    return 0;
                },
                opts);
        } else { // push
            plato::bc_opts_t opts;
            opts.local_capacity_ = 4 * PAGESIZE;
            // 广播计算得到新一轮激活结点数
            actives = plato::broadcast_message<plato::vid_t, plato::vid_t>(
                active_view,
                // PUSH消息发送函数
                [&](const plato::mepa_bc_context_t<plato::vid_t> &context,
                    plato::vid_t v_i) { 
                    // 发送当前master结点ID到其他分区的所有mirror结点
                    context.send(v_i); 
                },
                // PUSH消息接收函数
                [&](int /* p_i */, const plato::vid_t &v_i) {
                    plato::vid_t activated = 0;

                    auto neighbours = graph.first.neighbours(v_i);
                    // 遍历接收结点的(出度)邻接表
                    for (auto it = neighbours.begin_; neighbours.end_ != it;
                         ++it) {
                        plato::vid_t dst = it->neighbour_;
                        if ((parent[dst] == graph_info.vertices_) &&    // 无父结点
                            (plato::cas(&parent[dst], graph_info.vertices_,
                                        v_i))) {    // CAS设置父结点
                            active_next->set_bit(dst);
                            visited.set_bit(dst);
                            ++activated;
                        }
                    }
                    return activated;
                },
                opts);
        }

        std::swap(active_next, active_current);
    }

    if (0 == cluster_info.partition_id_) {
        LOG(INFO) << "bfs cost: " << watch.show("t2") / 1000.0 << "s";
    }

    // BFS访问的结点结果全局同步
    visited.sync();

    if (0 == cluster_info.partition_id_) {
        LOG(INFO) << "found vertices: " << visited.count()
                  << ", total cost: " << watch.show("t0") / 1000.0 << "s";
    }

    return 0;
}
