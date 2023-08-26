#ifndef __PLATO_ALGO_BP_HPP__
#define __PLATO_ALGO_BP_HPP__

#include <cstdint>
#include <cstdlib>
#include <cstdlib>
#include <type_traits>

#include "glog/logging.h"

#include "plato/util/perf.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/graph.hpp"
#include "plato/engine/dualmode.hpp"

namespace plato { namespace algo {

using bp_state_t = uint64_t;
using bp_dist_t  = double;

struct bp_factor_data_t {
  std::vector<std::pair<vid_t, bp_state_t>> vars_;
  std::vector<bp_dist_t> dists_;
};

struct bp_edata_t {
  uint32_t idx_;
  bp_dist_t* cur_msg_ = nullptr;
  bp_dist_t* next_msg_ = nullptr;
};


template <typename CACHE, typename  PART_IMPL, typename DATA, typename MSG>
int traverse_cache (
    CACHE& cache,
    const graph_info_t& graph_info,
    std::function<void(DATA*, bsp_send_callback_t<MSG>)> traverse_task,
    std::function<void(bsp_recv_pmsg_t<MSG>&)> recv_task,
    size_t chunk_size,
    const bsp_opts_t bsp_opts = bsp_opts_t(),
    const traverse_opts_t& traverse_opts = traverse_opts_t()
    ) {
  
  cache.reset_traversal(traverse_opts);
  auto __send = [&](bsp_send_callback_t<MSG> send) {
      while (cache.next_chunk([&](size_t, T* data) {
          traverse_task(data, send);
      }, &chunk_size)) {}
  };

  auto __recv = [&] (int /* p_i */, bsp_recv_pmsg_t<MSG>& pmsg) {
      recv_task(pmsg);
  };

  auto rc = fine_grain_bsp<MSG>(__send, __recv, bsp_opts);
  if (0 != rc) {
      LOG(ERROR) << "bsp failed with code: " << rc;
      return -1;
  }

  return 0;
}

template <template<typename> class VCACHE, typename  PART_IMPL, typename MSG>
int traverse_factors_cache (
    VCACHE<bp_factor_data_t>& pvacahe,
    const graph_info_t& graph_info,
    std::function<void(vertex_unit_t<bp_factor_data_t>*, bsp_send_callback_t<MSG>)> traverse_task,
    std::function<void(bsp_recv_pmsg_t<MSG>&)> recv_task
  ) {

  auto& cluster_info = cluster_info_t::get_instance();

  bsp_opts_t opts;
  opts.threads_               = -1;
  opts.flying_send_per_node_  = 3;
  opts.flying_recv_           = cluster_info.partitions_;
  opts.global_size_           = 16 * MBYTES;
  opts.local_capacity_        = PAGESIZE;
  opts.batch_size_            = 1;

  return traverse_cache(pvacahe, graph_info, traverse_task, recv_task, 1, opts);
}

class bp_bcsr_t : public bcsr_t<bp_edata_t, sequence_balanced_by_source_t> {
public:

  template <typename EDGE_CACHE>
  int load_from_ecache(const graph_info_t& graph_info, EDGE_CACHE& cache, bool is_outgoing = true) {

    auto reset_traversal = [&](bool auto_release_ = false) {
      traverse_opts_t trvs_opts; trvs_opts.mode_ = traverse_mode_t::RANDOM;
      trvs_opts.auto_release_ = auto_release_;
      cache.reset_traversal(trvs_opts);
    };

    auto foreach_srcs = [&](bsp_send_callback_t<vid_t> send) {
      auto traversal = [&](size_t, edge_unit_spec_t* edge) {
        CHECK(edge->src_ < vertices_);
        CHECK(edge->dst_ < vertices_);

        send(partitioner_->get_partition_id(edge->src_, edge->dst_), edge->src_);
        send(partitioner_->get_partition_id(edge->dst_, edge->src_), edge->dst_);
        return true;
      };

      size_t chunk_size = 64;
      while (cache.next_chunk(traversal, &chunk_size)) { }
    };

    auto foreach_edges = [&](bsp_send_callback_t<edge_unit_spec_t> send) {
      auto traversal = [&](size_t, edge_unit_spec_t* edge) {
        auto reversed_edge = *edge;     // an edge from variable to factor
        reversed_edge.src_ = edge->dst_;
        reversed_edge.dst_ = edge->src_;
        // only send the edges from factor to variable to record vidx in edata
        send(partitioner_->get_partition_id(edge->dst_, edge->src_), reversed_edge);
        return true;
      };

      size_t chunk_size = 64;
      while (cache.next_chunk(traversal, &chunk_size)) { }
    };

    int rc = -1;
    if (0 != (rc = load_from_traversal(graph_info.vertices_, reset_traversal, foreach_srcs, foreach_edges))) {
      return rc;
    }

    // TODO: foreach edges to send fidx from factor to var


  }

};







std::unique_ptr<thread_local_buffer> msg_buffer_p;

enum class bp_vertex_type_t {
  VARIABLE = 0,
  FACTOR = 1
};

struct bp_dist_size_msg_t {
  vid_t vid_;
  bp_vertex_type_t v_type_;
  bp_state_t dist_size_;
};

struct bp_var_dist_msg_t {
  uint32_t num_vars_;
  bp_state_t* var_dist_;

  template<typename Ar>
  void serialize(Ar &ar) {
    if (!var_dist_) {
      var_dist_ = (bp_state_t*)msg_buffer_p->local();
    }
    ar & num_vars_;
    for (uint32_t i = 0; i < num_vars_; ++i) {
      ar & var_dist_[i];
    }
  }
};

struct bp_vertex_state_t {
  bp_state_t dist_size_;
  bp_dist_t* dist_arr_;
  bp_dist_t* bp_arr_;
  bp_dist_t** msg_ptrs_arr_;

  bp_vertex_state_t(const bp_vertex_state_t&) = delete;
  bp_vertex_state_t& operator=(const bp_vertex_state_t&) = delete;
  bp_vertex_state_t(bp_vertex_state_t&& x) = default;
  bp_vertex_state_t& operator=(bp_vertex_state_t&& x) = default;
};


}}  // namespace algo, namespace plato

#endif