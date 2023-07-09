#ifndef __PLATO_ALGO_PERSONALIZED_PAGERANK_HPP__
#define __PLATO_ALGO_PERSONALIZED_PAGERANK_HPP__

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

struct personalized_pagerank_opts_t {
  uint32_t     iteration_ = 100;   // number of iterations
  double       damping_   = 0.85;  // the damping factor
  plato::vid_t src_       = 0;     // the source vertex
  double       eps_       = 0.001; // the calculation will be considered complete if the sum of
                                  // the difference of the 'rank' value between iterations 
                                  // changes less than 'eps'. if 'eps' equals to 0, pagerank will be
                                  // force to execute 'iteration' epochs.
};

/*
 * run personalized pagerank on a graph with incoming edges
 *
 * \tparam GRAPH  graph type, with incoming edges
 *
 * \param graph       the graph
 * \param graph_info  base graph-info
 * \param opts        personalized pagerank options
 *
 * \return
 *      each vertex's rank value in dense representation
 **/
template <typename GRAPH>
dense_state_t<double, typename GRAPH::partition_t> personalized_pagerank (
  GRAPH& graph,
  const graph_info_t& graph_info,
  const personalized_pagerank_opts_t& opts = personalized_pagerank_opts_t()) {

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

  vid_t src = opts.src_;
  rank_state_t curt_rank = engine.template alloc_v_state<double>();
  rank_state_t next_rank = engine.template alloc_v_state<double>();

  watch.mark("t1");
  auto odegrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, graph, false);
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate out-degrees from graph cost: " << watch.show("t1") / 1000.0 << "s";
  }

  double delta = curt_rank.template foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
        *pval = 0.0;
        return 1.0;
    }
  );
  // only initialize the rank of the src node to 1.
  curt_rank[src] = 1.0;
  if (odegrees[src] > 0) {
      curt_rank[src] = curt_rank[src] / odegrees[src];
  }

  // bool src_updated = false;
  // auto is_src_updated = [&](plato::vid_t v) -> bool {
  //   if(v == src && !src_updated) {
  //     return true;
  //   }
  //   return false;
  // }
  for (uint32_t epoch_i = 0; epoch_i < opts.iteration_; ++epoch_i) {
    watch.mark("t1");
  
    next_rank.fill(0.0);
    engine.template foreach_edges<double, int> (
      [&](const context_spec_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
        double rank_sum = 0.0;
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          rank_sum += curt_rank[it->neighbour_];
        }
        context.send(message_spec_t { v_i, rank_sum });
      },
      [&](int, message_spec_t& msg) {
        plato::write_add(&next_rank[msg.v_i_], msg.message_);
        return 0;
      }
    );

    if (opts.iteration_ - 1 == epoch_i) {
      delta = next_rank.template foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          // personalized
          *pval = (v_i == src ? 1.0 - opts.damping_ : 0.0) + opts.damping_ * (*pval);
          return 0;
        }
      );
    } else {
      delta = next_rank.template foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          // personalized
          *pval = (v_i == src ? 1.0 - opts.damping_ : 0.0) + opts.damping_ * (*pval);
          if (odegrees[v_i] > 0) {
            *pval = *pval / odegrees[v_i];
            return fabs(*pval - curt_rank[v_i]) * odegrees[v_i];
          }
          return fabs(*pval - curt_rank[v_i]);
        }
      );

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

}}

#endif