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

#ifndef __PLATO_ALGO_WEIGHTED_PAGERANK_HPP__
#define __PLATO_ALGO_WEIGHTED_PAGERANK_HPP__

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

struct pagerank_opts_t {
  uint32_t iteration_ = 100;   // number of iterations
  double   damping_   = 0.85;  // the damping factor
  double   eps_       = 0.001; // the calculation will be considered complete if the sum of
                              // the difference of the 'rank' value between iterations 
                              // changes less than 'eps'. if 'eps' equals to 0, pagerank will be
                              // force to execute 'iteration' epochs.
};

/*
 * run weighted_pagerank on a graph with incoming edges
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
dense_state_t<double, typename GRAPH::partition_t> weighted_pagerank (
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
  // 仅有DCSC的双模式引擎
  dualmode_engine_t<GRAPH, nullptr_t> engine (
      std::shared_ptr<GRAPH>(&graph, [](GRAPH*) {  }),
      nullptr,
      graph_info
  );

  /**
   * 有权图的PageRank算法说明:
   * 此算法是PageRank算法在有权图(即图中边有权重)的计算方法,并不是加权PageRank算法
   * 使用公式: PR(v_i)=(1-d)+d(\sum_{v_j\in InN(v_i)}(PR(v_j)*w_{v_j->v_i}/\sum_{v_k\in OutN(v_j)}w_{v_j->v_k}))
   * 其中:
   *  PR(v_i):即结点v_i的PageRank值
   *  d:阻尼系数,默认0.85
   *  InN(v_i):结点v_i的入边邻结点集
   *  OutN(v_i):结点v_i的出边邻结点集
   *  w_{v_j->v_i}:结点v_j到结点v_i的有向边(v_j,v_i)的权重
   * 
   * 在公式右边对结点v_i的入边邻结点的PageRank值计算时,
   *  β=w_{v_j->v_i}/\sum_{v_k\in OutN(v_j)}w_{v_j->v_k}
   * 对应无权图PageRank计算公式中的 1/{OutDeg(v_j)},即当边权重均相同时,二者相同
  */

  // 此轮迭代结点PageRank值稠密数据
  rank_state_t curt_rank = engine.template alloc_v_state<double>();
  // 下一轮迭代结点PageRank值稠密数据
  rank_state_t next_rank = engine.template alloc_v_state<double>();
  // 结点的出边权重之和,即\sum{v_k\in OutN(v_j)}w_{v_j->v_k}
  auto v_weight_sum = engine.template alloc_v_state<double>();

  watch.mark("t1");
  {
    auto e_traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
      // 遍历入边邻结点
      for (auto it = adjs.begin_; it != adjs.end_; it++) {
        // 累加出边权重
        write_add(&v_weight_sum[it->neighbour_], it->edata_);
      }
      return true;
    };
    graph.reset_traversal();
    #pragma omp parallel
    {
      size_t chunk_size = 1;
      while (graph.next_chunk(e_traversal, &chunk_size)) { }
    }
  }
  // 注:由于DCSC根据边的源结点划分master结点,因此master结点的出边都在一个集群节点中,
  // 所以无需全局归约
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate sum of weights from graph cost: " << watch.show("t1") / 1000.0 << "s";
  }

  // 两轮所有结点的PageRank值的差值之和
  double delta = curt_rank.template foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
      // 初始化每个结点PageRank值为1.
      *pval = 1.0;
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
          // β=当前边的权重/边源点的出边权重之和
          // 在无权图(即边权重均相同)的情况下,β=1/OutDeg(src),与无权图下的PageRank一致
          // β=w_{v_j->v_i}/\sum_{v_k\in OutN(v_j)}w_{v_j->v_k}
          // RP'(v_j)=RP(v_j)*β
          double beta = (it->edata_ * 1.0) / v_weight_sum[it->neighbour_];
          rank_sum += curt_rank[it->neighbour_] * beta;
        }
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
      delta = next_rank.template foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          // PR(v_i)=(1-d)+d(\sum_{v_j\in InN(v_i)}PR'(v_j))
          //        =(1-d)+d(\sum_{v_j\in InN(v_i)}(w_{(v_j->v_i)}/\sum_{v_k\in OutN(v_j)}w_{(v_j->v_k)}))
          *pval = 1.0 - opts.damping_ + opts.damping_ * (*pval);
          return 0;
        }
      );
    } else {  // 不为最后一轮迭代
      // 计算每个结点的PageRank值并返回差值
      delta = next_rank.template foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          *pval = 1.0 - opts.damping_ + opts.damping_ * (*pval);
          return fabs(*pval - curt_rank[v_i]);  // 结点v_i两轮PageRank的差值
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

