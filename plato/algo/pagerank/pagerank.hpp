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

#ifndef __PLATO_ALGO_PAGERANK_HPP__
#define __PLATO_ALGO_PAGERANK_HPP__

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato { namespace algo {

/// @brief PageRank选项
struct pagerank_opts_t {
  /// @brief 迭代次数 number of iterations
  uint32_t iteration_ = 100;
  /// @brief 阻尼系数 the damping factor
  double   damping_   = 0.85;
  /// @brief epsilon 收敛差值
  /// the calculation will be considered complete if the sum of
  /// the difference of the 'rank' value between iterations 
  /// changes less than 'eps'. if 'eps' equals to 0, pagerank will be
  /// force to execute 'iteration' epochs.
  double   eps_       = 0.001; 
};

/*
 * run pagerank on a graph with incoming edges
 *
 * \tparam GRAPH  graph type, with incoming edges
 *
 * \param graph       the graph
 * \param graph_info  base graph-info
 * \param opts        pagerank options
 *
 * \return
 *      each vertex's rank value in dense representation
 **/
template <typename GRAPH>
dense_state_t<double, typename GRAPH::partition_t> pagerank (
  GRAPH& graph,
  const graph_info_t& graph_info,
  const pagerank_opts_t& opts = pagerank_opts_t()) {

  using rank_state_t   = plato::dense_state_t<double, typename GRAPH::partition_t>;
  using context_spec_t = plato::mepa_ag_context_t<double>;
  using message_spec_t = plato::mepa_ag_message_t<double>;
  using adj_unit_list_spec_t = typename GRAPH::adj_unit_list_spec_t;

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  // init pull-only engine
  dualmode_engine_t<GRAPH, nullptr_t> engine (
      std::shared_ptr<GRAPH>(&graph, [](GRAPH*) {  }),
      nullptr,
      graph_info
  );

  /**
   * PageRank算法说明:
   * 使用公式: PR(A)=(1-d)+d(PR(T_1)/C(T_1)+...+PR(T_n)/C(T_n)) --Ref from Google
   * 其中:
   *  RP(A):结点A的PageRank值
   *  d:阻尼系数,默认0.85
   *  T_1,...,T_n:具有指向结点A的出边的结点,即结点A的入边邻结点
   *  C(A):结点A的出边数,即结点A的出度
   * 
   * 以下代码实现中会做一个预处理:
   * 每轮迭代计算PageRank值之后,会提前计算中间值PR'(A)=PR(A)/C(A),使得下一轮计算时可以直接使用,
   * 即原PageRank公式变为: PR(A)=(1-d)+d(PR'(T_1)+...+PR'(T_n))
   * 这也使得除去最后一轮迭代,cur_rank中记录的是PR'(A)=PR(A)/C(A)而非PR(A)
  */

  // 此轮迭代结点PageRank值稠密数据
  rank_state_t curt_rank = engine.template alloc_v_state<double>();
  // 下一轮迭代结点PageRank值稠密数据
  rank_state_t next_rank = engine.template alloc_v_state<double>();

  watch.mark("t1");
  // 结点出度稠密数据
  auto odegrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, graph, false);
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate out-degrees from graph cost: " << watch.show("t1") / 1000.0 << "s";
  }

  // 两轮所有结点的PageRank值的差值之和
  double delta = curt_rank.template foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
      // 初始化每个结点PageRank值为1.
      *pval = 1.0;
      if (odegrees[v_i] > 0) {
        // 为下一轮迭代计算PR'(A)=PR(A)/C(A)
        *pval = *pval / odegrees[v_i];
      }
      return 1.0;
    }
  );

  // 迭代指定轮数
  for (uint32_t epoch_i = 0; epoch_i < opts.iteration_; ++epoch_i) {
    watch.mark("t1");

    next_rank.fill(0.0);  // 初始化下一轮迭代PageRank值为0.
    // PULL模式遍历每条边
    engine.template foreach_edges<double, int> (
      // 消息发送函数
      [&](const context_spec_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
        // v_i入边邻结点此轮PageRank值(PR')之和
        double rank_sum = 0.0;
        // 遍历v_i的每个入边邻结点
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          // 累加邻结点的PR'值
          rank_sum += curt_rank[it->neighbour_];
        }
        // 发送结点及其入边邻结点PR'值之和
        context.send(message_spec_t { v_i, rank_sum });
      },
      // 消息接收函数
      [&](int, message_spec_t& msg) {
        // 累加v_i全局的入边邻结点PR'值之和
        plato::write_add(&next_rank[msg.v_i_], msg.message_);
        return 0;
      }
    );

    if (opts.iteration_ - 1 == epoch_i) { // 最后一轮迭代
      // 计算每个结点的PageRank值
      delta = next_rank.template foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          // PR(v_i)=(1-d)+d(\sum_{v_j\in InN(v_i)}PR'(v_j))=1-d+d(\sum_{v_j\in InN(v_i)}PR(v_j)/OutDeg(v_j))
          *pval = 1.0 - opts.damping_ + opts.damping_ * (*pval);
          return 0;
        }
      );
    } else {  // 不为最后一轮迭代
      // 计算每个结点的PageRank值并返回差值
      delta = next_rank.template foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          *pval = 1.0 - opts.damping_ + opts.damping_ * (*pval);
          // 为下一轮迭代预处理,即计算中间值PR'(v_i)=PR(v_i)/OutDeg(v_i)
          // 出度为0的结点不处理是因为没有结点会根据其计算PageRank值
          if (odegrees[v_i] > 0) {  
            *pval = *pval / odegrees[v_i];
            // 返回两轮PageRank值的差值(由于做了预处理因此需要再乘上出度)
            return fabs(*pval - curt_rank[v_i]) * odegrees[v_i];
          }
          return fabs(*pval - curt_rank[v_i]);
        }
      );
      // 若两轮PageRank的差值小于epsilon则再进行一轮迭代就终止
      if (opts.eps_ > 0.0 && delta < opts.eps_) {
        epoch_i = opts.iteration_ - 2;
      }
    }
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "], delta: " << delta << ", cost: "
        << watch.show("t1") / 1000.0 << "s";
    }
    std::swap(curt_rank, next_rank);
  }

  return curt_rank;
}

}}  // namespace algo, namespace pagerank

#endif

