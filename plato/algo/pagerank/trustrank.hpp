#ifndef __PLATO_ALGO_TRUSTRANK_HPP__
#define __PLATO_ALGO_TRUSTRANK_HPP__

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>

#include "glog/logging.h"
#include "boost/sort/sort.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"
#include "plato/algo/pagerank/pagerank.hpp"

namespace plato { namespace algo {

struct trustrank_opts_t {
  uint32_t iteration_     = 100;   // number of iterations
  double   damping_       = 0.85;  // the damping factor
  uint32_t seed_num_      = 100;   // number of seeds
  uint32_t select_method_ = 0;     // method of selecting seeds:
                                  // 0.PageRank; 1.Inverse PageRank; 2.Random
  double   eps_           = 0.001; // the calculation will be considered complete if the sum of
                                  // the difference of the 'rank' value between iterations 
                                  // changes less than 'eps'. if 'eps' equals to 0, pagerank will be
                                  // force to execute 'iteration' epochs.
};

struct rank_pair_t {
  vid_t v_i;
  double rank;
};

/*
 * generate a vector of rank pairs by PageRank
 *
 * \tparam GRAPH  graph type, with incoming edges or outgoing edges
 *
 * \param graph       the graph
 * \param graph_info  base graph-info
 *
 * \return
 *      each vertex's id and its rank value in a vector
 **/
template<typename GRAPH>
std::vector<rank_pair_t> generate_rank_pairs(GRAPH& graph, const graph_info_t& graph_info) {
  auto ranks = pagerank(graph, graph_info);

  sequence_v_view view = graph.partitioner()->self_v_view();
  std::vector<double> seed_ranks(graph_info.max_v_i_ + 1);
  #pragma omp parallel for
  for (vid_t v_i = view.start(); v_i < view.end(); ++v_i) {
    seed_ranks[v_i] = ranks[v_i];
  }
  allreduce(MPI_IN_PLACE, seed_ranks.data(), seed_ranks.size(), get_mpi_data_type<double>(), MPI_SUM, MPI_COMM_WORLD);
  
  std::vector<rank_pair_t> rank_pairs(graph_info.max_v_i_ + 1);
  #pragma omp parallel for
  for(vid_t v_i = 0; v_i < graph_info.max_v_i_ + 1; ++v_i) {
    rank_pairs[v_i] = rank_pair_t{ v_i, seed_ranks[v_i] };
  }

  return rank_pairs;
}

/*
 * select good seeds of TrustRank
 *
 * \tparam INCOMING      graph type, with incoming edges
 * \tparam OUTGOING      graph type, with outgoing edges
 *
 * \param in_edges       incoming edges, dcsc, ...
 * \param out_edges      outgoing edges, bcsr, ...
 * \param graph_info     base graph-info
 * \param good_vertices  a bitmap containing all good vertices
 * \param opts           trustrank options
 *
 * \return
 *    bitmap containing the selected good seeds
 * */
template <typename INCOMING, typename OUTGOING>
bitmap_t<> select_good_seeds(
  INCOMING& in_edges,
  OUTGOING& out_edges,
  const graph_info_t& graph_info,
  const bitmap_t<>& good_vertices,
  const trustrank_opts_t& opts) {

  bitmap_t<> good_seeds(graph_info.max_v_i_ + 1);
  
  // use good vertices as good seeds if number of vertices < seed_num_
  if (graph_info.max_v_i_ + 1 <= opts.seed_num_) {
    good_seeds.copy_from(good_vertices);
    return good_seeds;
  }

  if (opts.select_method_ != 2) {   // use PageRank or Inverse PageRank to select
    std::vector<rank_pair_t> rank_pairs = (opts.select_method_ == 0 ? 
                                          generate_rank_pairs(in_edges, graph_info) :
                                          generate_rank_pairs(out_edges, graph_info));

    boost::sort::pdqsort(rank_pairs.begin(), rank_pairs.end(), 
      [](const rank_pair_t& a, const rank_pair_t& b) {
        return a.rank > b.rank || (a.rank == b.rank && a.v_i < b.v_i);
      });

    // select good seed vertices from the top seed_num_ ranked nodes
    #pragma omp parallel for
    for (uint32_t i = 0; i < opts.seed_num_; ++i) {
      vid_t v_i = rank_pairs[i].v_i;
      // good seed must be a good vertex
      if (good_vertices.get_bit(v_i)) {
        good_seeds.set_bit(v_i);
      }
    }
  } else {  // select randomly
    // use the sum of seeds as the shared seed among processes to select the same random numbers
    unsigned local_rd = std::random_device()();
    unsigned rd_sum;
    MPI_Allreduce(&local_rd, &rd_sum, 1, get_mpi_data_type<unsigned>(), MPI_SUM, MPI_COMM_WORLD);

    std::default_random_engine generator(rd_sum);
    std::uniform_int_distribution<vid_t> distribution(0, graph_info.max_v_i_);

    std::unordered_set<vid_t> seed_set;   // record select seeds
    while (seed_set.size() < opts.seed_num_) {
      vid_t v_i = distribution(generator);
      if (seed_set.count(v_i) == 0) {
        seed_set.insert(v_i);
        // good seed must be a good vertex
        if (good_vertices.get_bit(v_i)) {
          good_seeds.set_bit(v_i);
        }
      }
    }
  }

  return good_seeds;
}

/*
 * run trustrank on a graph with incoming edges
 *
 * \tparam INCOMING      graph type, with incoming edges
 * \tparam OUTGOING      graph type, with outgoing edges
 *
 * \param in_edges       incoming edges, dcsc, ...
 * \param out_edges      outgoing edges, bcsr, ...
 * \param graph_info     base graph-info
 * \param good_vertices  a bitmap containing all good vertices
 * \param opts           trustrank options
 *
 * \return
 *    each vertex's rank value in dense representation
 * */
template <typename INCOMING, typename OUTGOING>
dense_state_t<double, typename INCOMING::partition_t> trustrank (
  INCOMING& in_edges,
  OUTGOING& out_edges,
  const graph_info_t& graph_info,
  const bitmap_t<>& good_vertices,
  const trustrank_opts_t& opts = trustrank_opts_t()) {

  using rank_state_t   = plato::dense_state_t<double, typename INCOMING::partition_t>;
  using context_spec_t = plato::mepa_ag_context_t<double>;
  using message_spec_t = plato::mepa_ag_message_t<double>;
  using adj_unit_list_spec_t = typename INCOMING::adj_unit_list_spec_t;
  using v_subset_t    = bitmap_t<>;

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  dualmode_engine_t<INCOMING, OUTGOING> engine (
    std::shared_ptr<INCOMING>(&in_edges,  [](INCOMING*) { }),
    std::shared_ptr<OUTGOING>(&out_edges, [](OUTGOING*) { }),
    graph_info);

  watch.mark("t1");
  auto odegrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, in_edges, false);
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate out-degrees from graph cost: " << watch.show("t1") / 1000.0 << "s";
  }
  
  // select good seeds by opts.select_method
  v_subset_t good_seeds = select_good_seeds(in_edges, out_edges, graph_info, good_vertices, opts);

  rank_state_t curt_rank = engine.template alloc_v_state<double>();
  rank_state_t next_rank = engine.template alloc_v_state<double>();

  double init_dist = 1. / good_seeds.count();
  double delta = curt_rank.template foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
      *pval = (good_seeds.get_bit(v_i) ? init_dist : 0.0);
      if (odegrees[v_i] > 0) {
        *pval = *pval / odegrees[v_i];
      }
      return 1.0;
    }
  );
  
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
          // use good seeds to compute
          *pval = (good_seeds.get_bit(v_i) ? (1.0 - opts.damping_) * init_dist : 0.0) + opts.damping_ * (*pval);
          return 0;
        }
      );
    } else {
      delta = next_rank.template foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          // use good seeds to compute
          *pval = (good_seeds.get_bit(v_i) ? (1.0 - opts.damping_) * init_dist : 0.0) + opts.damping_ * (*pval);
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