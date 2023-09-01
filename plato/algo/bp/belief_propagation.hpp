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

struct bp_opts_t {
  uint32_t iteration_ = 100;   // number of iterations
  double   eps_       = 0.001; // the calculation will be considered complete if the sum of
                              // the difference of the 'rank' value between iterations 
                              // changes less than 'eps'. if 'eps' equals to 0, pagerank will be
                              // force to execute 'iteration' epochs.
};

using bp_dist_size_t = uint64_t;
using bp_prob_t      = double;

struct bp_factor_data_t {
  std::vector<std::pair<vid_t, bp_dist_size_t>> vars_;
  std::vector<bp_prob_t> dists_;
};

struct bp_edata_t {
  uint32_t       idx_;
  bp_dist_size_t msg_offset_ = 0;

  template<typename Ar>
  void serialize(Ar &ar) {
    ar & idx_;
  }
};


template <typename CACHE, typename  PART_IMPL, typename DATA, typename MSG>
int traverse_cache_bsp (
    CACHE& cache,
    std::function<void(DATA*, bsp_send_callback_t<MSG>)> traverse_task,
    std::function<void(bsp_recv_pmsg_t<MSG>&)> recv_task,
    size_t chunk_size,
    const bsp_opts_t bsp_opts = bsp_opts_t(),
    const traverse_opts_t& traverse_opts = traverse_opts_t()) {
  
  cache.reset_traversal(traverse_opts);
  auto __send = [&](bsp_send_callback_t<MSG> send) {
    size_t local_chunk_size = chunk_size;
    while (cache.next_chunk([&](size_t, T* data) {
        traverse_task(data, send);
      }, &local_chunk_size)) {}
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
int traverse_factors_cache_bsp (
    VCACHE<bp_factor_data_t>& pvacahe,
    std::function<void(vertex_unit_t<bp_factor_data_t>*, bsp_send_callback_t<MSG>)> traverse_task,
    std::function<void(bsp_recv_pmsg_t<MSG>&)> recv_task) {

  auto& cluster_info = cluster_info_t::get_instance();

  bsp_opts_t opts;
  opts.threads_               = -1;
  opts.flying_send_per_node_  = 3;
  opts.flying_recv_           = cluster_info.partitions_;
  opts.global_size_           = 16 * MBYTES;
  opts.local_capacity_        = PAGESIZE;
  opts.batch_size_            = 1;

  return traverse_cache_bsp(pvacahe, traverse_task, recv_task, 1, opts);
}

struct bp_dist_t {
  bp_dist_size_t size_   = 0;
  bp_prob_t*     values_ = nullptr;

  bp_dist_t() = default;
  bp_dist_t(bp_prob_t* values, bp_dist_size_t size): values_(values), size_(size) {}

  bp_dist_t(const bp_dist_t&) = delete;
  bp_dist_t& operator=(const bp_dist_t&) = delete;
};

template <typename PART_IMPL, typename ALLOC = mmap_allocator_t<bp_dist_size_t>, typename BITMAP = bitmap_t<>>
class bp_dense_dists_t: public dense_state_t<bp_dist_size_t, PART_IMPL, ALLOC, BITMAP> {
public:
  bp_dense_dists_t(const bp_dist_size_t* dist_offsets, vid_t max_v_i, std::shared_ptr<partition_t> partitioner) : 
      dense_state_t<bp_dist_size_t, PART_IMPL, ALLOC, BITMAP>(max_v_i + 1, partitioner), buf_allocator_() {
    dist_buf_ = buf_allocator_.allocate(dist_offsets[max_v_i + 1]);
    std::fill(dist_buf_, dist_buf_ + dist_offsets[max_v_i + 1], 1.0);
    
    #pragma omp parallel for
    for(vid_t i = 0; i <= max_v_i + 1; ++i) {
      data_[i] = dist_offsets[i];
    }
  }

  ~bp_dense_dists_t() {
    if (nullptr != dist_buf_) {
      buf_allocator_.deallocate(dist_buf_, data_[max_v_i_ - 1]);
    }
  }

private:
  using buf_traits_ = typename std::allocator_traits<ALLOC>::template rebind_traits<bp_prob_t>;
  using buf_allocator_type = typename buf_traits_::allocator_type;

  buf_allocator_type buf_allocator_;
  // bp_dist_size_t dist_buf_size_;
  bp_prob_t* dist_buf_;
};

enum class bp_vertex_type_t {
  VARIABLE = 0,
  FACTOR   = 1
};

template <typename ALLOC = std::allocator<adj_unit_t<bp_edata_t>>>
class bp_bcsr_t : public bcsr_t<bp_edata_t, sequence_balanced_by_source_t, ALLOC> {
public:

  /*
   *
   * \param from_factor  true  -- factor-to-variable edges | false -- variable-to-factor edges
   * \param is_vidx      true  -- the idx_ field of bp_edata_t records the variable's index
   *                              in the factor's adjacency list (vidx) | 
   *                     false -- records the factor's index in the variable's adjacency list (fidx)
   * */
  template <typename ECACHE>
  int load_from_edges_cache(const graph_info_t& graph_info, ECACHE& pecache, bool from_factor = true, bool is_vidx = true) {

    auto reset_traversal = [&](bool auto_release_ = false) {
      traverse_opts_t trvs_opts;
      trvs_opts.mode_ = traverse_mode_t::RANDOM;
      trvs_opts.auto_release_ = auto_release_;
      pecache.reset_traversal(trvs_opts);
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
      while (pecache.next_chunk(traversal, &chunk_size)) { }
    };

    auto foreach_edges = [&](bsp_send_callback_t<edge_unit_spec_t> send) {
      auto traversal = [&](size_t, edge_unit_spec_t* edge) {
        // only send the variable-to-factor edges with vidx or factor-to-varible edges with fidx
        if (from_factor == is_vidx) {
          auto reversed_edge = *edge;
          reversed_edge.src_ = edge->dst_;
          reversed_edge.dst_ = edge->src_;
          send(partitioner_->get_partition_id(edge->dst_, edge->src_), reversed_edge);
        } else {
          send(partitioner_->get_partition_id(edge->src_, edge->dst_), *edge);
        }
        return true;
      };

      size_t chunk_size = 64;
      while (pecache.next_chunk(traversal, &chunk_size)) { }
    };

    int rc = -1;
    if (0 != (rc = load_from_traversal(graph_info.vertices_, reset_traversal, foreach_srcs, foreach_edges))) {
      return rc;
    }

    // TODO: foreach variable-to-factor/factor-to-variable edges
    aggregate_message<std::pair<vid_t, uint32_t>, int, bp_bcsr_t>(*this, 
      [&](mepa_ag_context_t<std::pair<vid_t, uint32_t>>& context, vid_t v_i, const adj_unit_list_spec_t& adjs) {
        for (uint32_t idx = 0; idx < adjs.end_ - adjs.begin_; ++idx) {
          auto it = adjs.begin_ + idx;
          // a pair of (varible, fidx) / (factor, vidx)
          std::pair<vid_t, uint32_t> edge_pair = std::make_pair(v_i, idx);
          // send an factor-to-variable/variable-to-factor edge with fidx/vidx to factor/variable
          context.send(mepa_ag_message_t<std::pair<vid_t, uint32_t>> {it->neighbour_, edge_pair});
        }
      },
      [&](int, mepa_ag_message_t<std::pair<vid_t, uint32_t>>& msg) {
        eid_t idx = __sync_fetch_and_add(&index_.get()[msg.v_i_+1], (eid_t)1);
        auto& nei = adjs_.get()[idx];
        nei.neighbour_  = msg.message_.first;   // varible/factor id
        nei.edata_.idx_ = msg.message_.second;  // fidx/vidx
      });

    return 0;
  }

  template <typename VCACHE>
  int init_states_from_factors_cache(VCACHE& pvcache) {

    {
      bitmap_allocator_t __alloc(allocator_);
      auto* __p = __alloc.allocate(1);
      __alloc.construct(__p, vertices_);

      local_factors_map_.reset(__p, [__alloc](bitmap_pointer p) mutable {
        __alloc.destroy(p);
        __alloc.deallocate(p, 1);
      });
    }

    std::shared_ptr<bp_dist_size_t> dist_sizes;
    {
      dist_sizes_allocator_t __alloc(allocator_);
      auto* __p = __alloc.allocate(vertices_);
      memset(__p, 0, sizeof(bp_dist_size_t) * vertices_);

      dist_sizes.reset(__p, [__alloc, vertices_](dist_sizes_pointer p) mutable {
        __alloc.deallocate(p, vertices_);
      });
    }

    // init local_factors_map_ and dist_sizes
    {
      // local_dist_size_ = 0;
      plato::bitmap_t<> local_var_map(vertices_);
      traverse_factors_cache_bsp(pvcache,
        [&](plato::vertex_unit_t<bp_factor_data_t>* v_data, plato::bsp_send_callback_t<dist_size_msg_t> send) {
          auto var_size = v_data->vdata_.vars_.size();
          if (var_size > 1) {
            dist_size_msg_t factor_msg;
            factor_msg.v_i_       = v_data->v_i_;
            factor_msg.v_type_    = bp_vertex_type_t::FACTOR;
            factor_msg.dist_size_ = v_data->vdata_.dists_.size();
            send(partitioner_->get_partition_id(factor_msg.v_i_), factor_msg);
          }
          for(const auto& var: v_data->vdata_.vars_) {
            dist_size_msg_t var_msg;
            var_msg.v_i_       = var.first;
            var_msg.v_type_    = bp_vertex_type_t::VARIABLE;
            var_msg.dist_size_ = var.second;
            send(partitioner_->get_partition_id(var_msg.v_i_), var_msg);
          }
        },
        [&](plato::bsp_recv_pmsg_t<dist_size_msg_t>& pmsg) {
          plato::vid_t v_i = pmsg->v_i_;
          CHECK(bitmap_->get_bit(v_i)) << "vertex " << v_i << " is not in the graph";
          if (pmsg->v_type_ == bp_vertex_type_t::VARIABLE) {
            CHECK(local_factors_map_->get_bit(v_i) == 0) << "vertex " << v_i << " is a factor vertex";
            if (local_var_map.get_bit(v_i) != 0) {
              return;
            }
            local_var_map.set_bit(v_i);
          } else {
            CHECK(local_var_map.get_bit(v_i) == 0) << "vertex " << v_i << " is a variable vertex";
            CHECK(local_factors_map_->get_bit(v_i) == 0) << "duplicated factor vertex: " << v_i;
            local_factors_map_->set_bit(v_i);
          }
          dist_sizes.get()[v_i] = pmsg->dist_size_;
          // __sync_fetch_and_add(&local_dist_size_, pmsg->dist_size_);
        });
    }
    allreduce(MPI_IN_PLACE, dist_sizes.get(), vertices_, get_mpi_data_type<bp_dist_size_t>(), MPI_SUM, MPI_COMM_WORLD);

    {
      dist_sizes_allocator_t __alloc(allocator_);
      auto* __p = __alloc.allocate(vertices_ + 1);
      memset(__p, 0, sizeof(bp_dist_size_t) * (vertices_ + 1));

      dist_offsets_.reset(__p, [__alloc, vertices_](dist_sizes_pointer p) mutable {
        __alloc.deallocate(p, vertices_ + 1);
      });
    }

    #pragma omp parallel for
    for (vid_t v_i = 1; v_i <= vertices_; ++v_i) {
      dist_offsets_.get()[v_i] = dist_sizes.get()[v_i - 1];
    }

    for (vid_t v_i = 1; v_i <= vertices_; ++v_i) {
      dist_offsets_.get()[v_i] += dist_offsets_.get()[v_i - 1];
    }
    // {
    //   dist_ptr_allocator_t __alloc(allocator_);
    //   auto* __p = __alloc.allocate(vertices_);
    //   memset(__p, 0, sizeof(bp_prob_t*) * vertices_);

    //   dists_.reset(__p, [__alloc, vertices_](dist_ptr_pointer p) mutable {
    //     __alloc.deallocate(p, vertices_);
    //   });
    // }

    {
      dist_buf_allocator_t __alloc(allocator_);
      bp_dist_size_t __local_dist_size = dist_offsets_.get()[vertices_];
      auto* __p = __alloc.allocate(__local_dist_size);
      // initialize the probability distribution to all 1.0
      std::fill(__p, __p + __local_dist_size, bp_prob_t(1.0));

      dists_buf_.reset(__p, [__alloc, __local_dist_size](dist_buf_pointer p) mutable {
        __alloc.deallocate(p, __local_dist_size);
      });
    }

    // // init ptr of dists_ to dists_buf_
    // {
    //   bp_dist_size_t dist_idx = 0;
    //   #pragma omp parallel for
    //   for (vid_t v_i = 0; v_i < vertices_; ++v_i) {
    //     if (bitmap_->get_bit(v_i)) {
    //       dists_.get()[v_i] = dists_buf_.get() + __sync_fetch_and_add(&dist_idx, dist_sizes_.get()[v_i]);
    //     }
    //   }
    //   CHECK(dist_idx == local_dist_size_);
    // }

    // init values of dists_
    {
      msg_bsp_buf_p_.reset(new thread_local_buffer);
      auto msg_buf_reset_defer = plato::defer([msg_bsp_buf_p_] { msg_bsp_buf_p_.reset(); });
      traverse_factors_cache_bsp(pvcache, 
        [&] (plato::vertex_unit_t<bp_factor_data_t>* v_data, plato::bsp_send_callback_t<dist_msg_t> send) {
          dist_msg_t msg;
          msg.size_ = v_data->vdata_.dists_.size();
          msg.dist_ = v_data->vdata_.dists_.data();
          if (v_data->vdata_.vars_.size() > 1) {
            msg.v_i_ = v_data->vid_;
          } else {
            msg.v_i_ = v_data->vdata_.vars_.front().first;
          }
          send(partitioner_->get_partition_id(msg.v_i_), msg);
        },
        [&](plato::bsp_recv_pmsg_t<dist_msg_t>& pmsg) {
          vid_t v_i = pmsg->v_i_;
          CHECK(pmsg->size_ == dist_offsets_.get()[v_i + 1] - dist_offsets_.get()[v_i]);
          memcpy(dists_buf_.get() + dist_offsets_.get()[v_i], pmsg->dist_, sizeof(bp_prob_t) * pmsg->size_);
        });
    }

    std::unique_ptr<bp_dist_size_t> msg_offsets;
    {
      dist_sizes_allocator_t __alloc(allocator_);
      auto* __p = __alloc.allocate(vertices_ + 1);
      memset(__p, 0, sizeof(bp_dist_size_t) * (vertices_ + 1));

      msg_offsets.reset(__p, [__alloc, vertices_](dist_sizes_pointer p) mutable {
        __alloc.deallocate(p, vertices_ + 1);
      });
    }

    #pragma omp parallel for
    for (vid_t v_i = 0; v_i < vertices_; ++v_i) {
      if (bitmap_->get_bit(v_i) == 0) {
        continue;
      }
      bp_dist_size_t msg_size = 0;
      if (local_factors_map_->get_bit(v_i)) {
          for (eid_t idx = index_.get()[v_i]; idx < index_.get()[v_i + 1]; ++idx) {
            vid_t vid = adjs_.get()[idx].neighbour_;
            msg_size += dist_sizes.get()[vid];
          }
      } else {
        msg_size += dist_sizes.get()[v_i] * (index_.get()[v_i + 1] - index_.get()[v_i]);
      }
      msg_offsets.get()[v_i + 1] = msg_size;
    }

    for (vid_t v_i = 1; v_i <= vertices_; ++v_i) {
      msg_offsets.get()[v_i] += msg_offsets.get()[v_i - 1];
    }
    msg_buf_size_ = msg_offsets.get()[vertices_];

    #pragma omp parallel for
    for (vid_t v_i = 0; v_i < vertices_; ++v_i) {
      if (bitmap_->get_bit(v_i) == 0) {
        continue;
      }
      bp_dist_size_t offset = msg_offsets.get()[v_i];
      if (local_factors_map_->get_bit(v_i)) {
        for (eid_t idx = index_.get()[v_i]; idx < index_.get()[v_i + 1]; ++idx) {
          auto& edge = adjs_.get()[idx];
          edge.edata_.msg_offset_ = offset;
          offset += dist_sizes.get()[edge.neighbour_];
        }
      } else {
        for (eid_t idx = index_.get()[v_i]; idx < index_.get()[v_i + 1]; ++idx) {
          auto& edge = adjs_.get()[idx];
          edge.edata_.msg_offset_ = offset;
          offset += dist_sizes.get()[v_i];
        }
      }
    }

    return 0;
  }
  
  template <typename ECACHE, typename VCACHE> 
  int load_from_cache(const graph_info_t& graph_info, ECACHE& pecache, VCACHE& pvcache, bool from_factor = true, bool is_vidx = true) {
    int rc = -1;
    if (0 != (rc = load_from_edges_cache(graph_info, pecache, from_factor, is_vidx))) {
      return rc;
    }
    return init_states_from_factors_cache(pvcache);
  }

  
  bp_dense_dists_t<partition_t> alloc_dist_state() const {
    return bp_dense_dists_t<partition_t>(dist_offsets_.get(), vertices_ - 1, partitioner_);
  }

  std::shared_ptr<bp_prob_t> alloc_msg_buf() const {
    dist_buf_allocator_t __alloc(allocator_);
    auto* __p = __alloc.allocate(msg_buf_size_);
    // initialize the probability distribution to all 1.0
    std::fill(__p, __p + msg_buf_size_, bp_prob_t(1.0));

    std::shared_ptr<bp_prob_t> msg_buf(__p, [__alloc, msg_buf_size_](dist_buf_pointer p) mutable {
      __alloc.deallocate(p, msg_buf_size_);
    });
    return msg_buf;
  }

private:
  using traits_ = typename std::allocator_traits<ALLOC>::template rebind_traits<adj_unit_t<edata_t>>;

protected:
  using dist_sizes_allocator_t = typename traits_::template rebind_alloc<bp_dist_size_t>;
  using dist_sizes_traits_     = typename traits_::template rebind_traits<bp_dist_size_t>;
  using dist_sizes_pointer     = typename dist_sizes_traits_::pointer;

  // using dist_ptr_allocator_t = typename traits_::template rebind_alloc<bp_prob_t*>;
  // using dist_ptr_traits_     = typename traits_::template rebind_traits<bp_prob_t*>;
  // using dist_ptr_pointer     = typename dist_ptr_traits_::pointer;

  using dist_buf_allocator_t = typename traits_::template rebind_alloc<bp_prob_t>;
  using dist_buf_traits_     = typename traits_::template rebind_traits<bp_prob_t>;
  using dist_buf_pointer     = typename dist_buf_traits_::pointer;

  std::shared_ptr<bitmap_spec_t>  local_factors_map_;
  std::shared_ptr<bp_dist_size_t> dist_offsets_;
  std::shared_ptr<bp_prob_t>      dists_buf_;

  bp_dist_size_t msg_buf_size_;

  // bp_dist_size_t local_dist_size_;
  // std::shared_ptr<bp_dist_size_t> dist_sizes_;
  // std::shared_ptr<bp_prob_t*>     dists_;


private:
  static std::unique_ptr<plato::thread_local_buffer> msg_bsp_buf_p_;

  struct dist_size_msg_t {
    vid_t v_i_;
    bp_vertex_type_t v_type_;
    bp_dist_size_t dist_size_;
  };

  struct dist_msg_t {
    vid_t          v_i_;
    bp_dist_size_t size_ = 0;
    bp_prob_t*     dist_ = nullptr;

    template<typename Ar>
    void serialize(Ar &ar) {
      if (!dist_) {
        dist_ = (bp_prob_t*)msg_bsp_buf_p_->local();
      }
      ar & v_i_;
      ar & size_;
      for (uint32_t i = 0; i < size_; ++i) {
        ar & dist_[i];
      }
    }
  };
};

template <typename ALLOC>
bp_dense_dists_t<typename bp_bcsr_t<ALLOC>::partition_t> belief_propagation (
    bp_bcsr_t<ALLOC>& graph,
    const graph_info_t& graph_info,
    const bp_opts_t& opts = bp_opts_t()) {
  
  using belief_state_t = bp_dense_dists_t<typename bp_bcsr_t<ALLOC>::partition_t>;
  using adj_unit_list_spec_t = typename bp_bcsr_t<ALLOC>::adj_unit_list_spec_t;

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  belief_state_t curt_belief = graph.alloc_dist_state();
  belief_state_t next_belief = graph.alloc_dist_state();
  auto curt_msgs = graph.alloc_msg_buf();
  auto next_msgs = graph.alloc_msg_buf();

  double delta = curt_belief.template foreach<double> (
    [&] (plato::vid_t v_i, bp_dist_t* dval) {
      return 1.0 * dval->size_;
    }
  )



}

}}  // namespace algo, namespace plato

#endif