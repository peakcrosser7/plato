#ifndef __PLATO_ALGO_TRUSTRANK_HPP__
#define __PLATO_ALGO_TRUSTRANK_HPP__

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <numeric> 
#include <algorithm>

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
 * select good seeds of TrustRank
 *
 * \tparam INCOMING      graph type, with incoming edges
 * \tparam OUTGOING      graph type, with outgoing edges
 *
 * \param in_edges       incoming edges, dcsc, ...
 * \param out_edges      outgoing edges, bcsr, ...
 * \param graph_info     base graph-info
 * \param good_vertices  a bitmap containing good vertices
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
  trustrank_opts_t& opts) {

  bitmap_t<> good_seeds(graph_info.max_v_i_ + 1);
  
  // use good vertices as good seeds if number of vertices < seed_num_
  if (graph_info.max_v_i_ + 1 <= opts.seed_num_) {
    opts.seed_num_ = graph_info.max_v_i_ + 1;
    good_seeds.copy_from(good_vertices);
    return good_seeds;
  }

  if (opts.select_method_ != 2) {   // use PageRank or Inverse PageRank to select
    std::vector<rank_pair_t> rank_pairs(graph_info.max_v_i_ + 1);
    if (opts.select_method_ == 0) {
      auto seed_ranks = pagerank<INCOMING>(in_edges, graph_info);
      seed_ranks.template foreach<int>(
        [&](vid_t v_i, double* pval) {
          rank_pairs[v_i] = rank_pair_t{ v_i, *pval };
          return 0;
        }
      );
    } else {
      auto seed_ranks = pagerank<OUTGOING>(out_edges, graph_info);
      seed_ranks.template foreach<int>(
        [&](vid_t v_i, double* pval) {
          rank_pairs[v_i] = rank_pair_t{ v_i, *pval };
          return 0;
        }
      );
    }

    boost::sort::pdqsort(rank_pairs.begin(), rank_pairs.end(), 
      [](const rank_pair_t& a, const rank_pair_t& b) {
        return a.rank > b.rank || (a.rank == b.rank && a.v_i < b.v_i);
      });

    // select good seed vertices from the top seed_num_ ranked nodes
    for (uint32_t i = 0; i < opts.seed_num_; ++i) {
      vid_t v_i = rank_pairs[i].v_i;
      // good seed must be a good vertex
      if (good_vertices.get_bit(v_i)) {
        good_seeds.set_bit(v_i);
      }
    }
  } else {  // select randomly
    sequence_v_view view = in_edges.partitioner()->self_v_view();
    vid_t start = view.start(), end = view.end();
    std::vector<vid_t> vertices(end - start);
    std::iota(vertices.begin(), vertices.end(), start);
    // shuffle vertices the partition owned
    std::random_shuffle(vertices.begin(), vertices.end());

    auto& cluster_info = plato::cluster_info_t::get_instance();
    uint32_t partitions = cluster_info.partitions_, pid = cluster_info.partition_id_;
    std::vector<vid_t> seeds(opts.seed_num_, 0);
    std::vector<uint32_t> seed_counts(partitions, 0);

    uint32_t psize = opts.seed_num_ / partitions;
    // std::min() to avoid the partition with fewer vertices than to be selected
    uint32_t count = std::min(end - start, psize + (pid < (opts.seed_num_ % partitions)));
    uint32_t offset = psize * pid + std::min(pid, opts.seed_num_ % partitions);
    // select "count" vertices as seeds in the partition
    std::copy(vertices.begin(), vertices.begin() + count, seeds.begin() + offset); 
    seed_counts[pid] = count;

    // synchronize the selected seeds and its count for each partition
    allreduce(MPI_IN_PLACE, seeds.data(), seeds.size(), get_mpi_data_type<vid_t>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, seed_counts.data(), seed_counts.size(), get_mpi_data_type<uint32_t>(), MPI_SUM, MPI_COMM_WORLD);
    
    uint32_t seed_num = std::accumulate(seed_counts.begin(), seed_counts.end(), 0U);
    if (seed_num == opts.seed_num_) {
      for(vid_t v_i: seeds) {
        if (good_vertices.get_bit(v_i)) {
          good_seeds.set_bit(v_i);
        }
      }
    } else {  // when the number of selected seeds in some partitions is not enough
      uint32_t offset = 0;
      for (uint32_t p = 0; p < partitions; ++p) {
        for (uint32_t i = 0; i < seed_counts[p]; ++i) {
          vid_t v_i = seeds[offset + i];
          if (good_vertices.get_bit(v_i)) {
            good_seeds.set_bit(v_i);
          }
        }
        offset += psize + (p < (opts.seed_num_ % partitions));
      }
      opts.seed_num_ = seed_num;
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
 * \param good_vertices  a bitmap containing good vertices
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
  trustrank_opts_t& opts = trustrank_opts_t()) {

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

  double init_dist = 1./opts.seed_num_;
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