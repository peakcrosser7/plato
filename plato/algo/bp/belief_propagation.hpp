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
                              // the difference of the 'belief' value between iterations 
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

struct bp_dist_t {
  bp_dist_size_t size_   = 0;
  bp_prob_t*     values_ = nullptr;

  bp_dist_t() = default;
  bp_dist_t(bp_prob_t* values, bp_dist_size_t size): size_(size), values_(values) {}

  void fill(bp_prob_t value) {
    if (values_ != nullptr) {
      for (bp_dist_size_t i = 0; i < size_; ++i) {
        values_[i] = value;
      }
    }
  }

  void multiply(const bp_dist_t& rhs) {
    CHECK(values_ && rhs.values_);
    CHECK(size_ == rhs.size_);
    for (bp_dist_size_t i = 0; i < size_; ++i) {
      values_[i] *= rhs.values_[i];
    }   
  }

  void normalize() {
    CHECK(values_);
    bp_prob_t sum = 0.0;
    for (bp_dist_size_t i = 0; i < size_; ++i) {
      sum += values_[i];
    }
    if (sum == 0.0) {
      return;
    }
    for (bp_dist_size_t i = 0; i < size_; ++i) {
      values_[i] /= sum;
    }
  }

  void clean() {
    size_ = 0;
    values_ = nullptr;
  }

  std::string to_string() const {
    std::string str;
    if (!values_ || size_ == 0) {
      str += "";
    } else {
      str += "[";
      for (bp_dist_size_t i = 0; i < size_; ++i) {
        str += std::to_string(values_[i]) + " ";
      }
      str += "]";
    }
    return str;
  }

  friend std::ostream& operator<<(std::ostream& os, const bp_dist_t& rhs);
};

template <typename PART_IMPL, typename ALLOC = mmap_allocator_t<bp_dist_t>, typename BITMAP = bitmap_t<>>
class bp_dense_dists_t : public dense_state_t<bp_dist_t, PART_IMPL, ALLOC, BITMAP> {
public:
  void fill(bp_prob_t value) {
    #pragma omp parallel for
    for (bp_dist_size_t i = 0; i < dist_buf_size_; ++i) {
      dist_buf_[i] = value;
    }
  }

  bp_dense_dists_t(const bp_dist_size_t* dist_offsets, vid_t max_v_i, std::shared_ptr<PART_IMPL> partitioner);

  bp_dense_dists_t(bp_dense_dists_t&& other);
  bp_dense_dists_t& operator=(bp_dense_dists_t&& other);

  bp_dense_dists_t(const bp_dense_dists_t&) = delete;
  bp_dense_dists_t& operator=(const bp_dense_dists_t&) = delete;

  ~bp_dense_dists_t() {
    if (nullptr != dist_buf_) {
      buf_allocator_.deallocate(dist_buf_, dist_buf_size_);
    }
  }

private:
  using buf_traits_ = typename std::allocator_traits<ALLOC>::template rebind_traits<bp_prob_t>;
  using buf_allocator_type = typename buf_traits_::allocator_type;

  buf_allocator_type buf_allocator_;
  bp_dist_size_t     dist_buf_size_;
  bp_prob_t*         dist_buf_;
};

enum class bp_vertex_type_t {
  VARIABLE = 0,
  FACTOR   = 1
};

template <typename ALLOC = std::allocator<adj_unit_t<bp_edata_t>>>
class bp_bcsr_t : public bcsr_t<bp_edata_t, sequence_balanced_by_source_t, ALLOC> {
private:
  static std::unique_ptr<plato::thread_local_buffer> bsp_msg_buf_p_;

public:
  using bcsr_spec_t = bcsr_t<bp_edata_t, sequence_balanced_by_source_t, ALLOC>;
  using bitmap_spec_t = typename bcsr_spec_t::bitmap_spec_t;

  // ******************************************************************************* //
  // required types & methods

  using edata_t              = typename bcsr_spec_t::edata_t;
  using partition_t          = typename bcsr_spec_t::partition_t;
  using allocator_type       = typename bcsr_spec_t::allocator_type;
  using edge_unit_spec_t     = typename bcsr_spec_t::edge_unit_spec_t;
  using adj_unit_spec_t      = typename bcsr_spec_t::adj_unit_spec_t;
  using adj_unit_list_spec_t = typename bcsr_spec_t::adj_unit_list_spec_t;

  struct bp_msg_t {
    // vid_t     to_;
    uint32_t  from_idx_;
    bp_dist_t msg_;

    template<typename Ar>
    void serialize(Ar &ar) {
      if (!msg_.values_) {
        msg_.values_ = (bp_prob_t*)bsp_msg_buf_p_->local();
      }
      // ar & to_;
      ar & from_idx_;
      ar & msg_.size_;
      for (uint32_t i = 0; i < msg_.size_; ++i) {
        ar & msg_.values_[i];
      }
    }
  };

  /*
   * parallel load edges from edges' cache
   *
   * \tparam ECACHE       edges' cache type
   *
   * \param  graph_info   graph info
   * \param  pecache      edges' cache
   * \param  from_factor  true  -- factor-to-variable edges | false -- variable-to-factor edges
   * \param  is_vidx      true  -- the 'idx_' field of 'bp_edata_t' records the variable's index in
   *                               the factor's adjacency list (vidx) | 
   *                      false -- records the factor's index in the variable's adjacency list (fidx)
   * 
   * \return 0 -- success, else failed
   **/
  template <typename ECACHE>
  int load_from_edges_cache(const graph_info_t& graph_info, ECACHE& pecache, bool from_factor = true, 
      bool is_vidx = true);

  /*
   * initialize states of the factor graph from factor vertices' cache
   *
   * \tparam VCACHE       vertices' cache type
   *
   * \param  pvcache      factor vertices' cache
   * 
   * \return 0 -- success, else failed
   **/
  template <typename VCACHE>
  int init_states_from_factors_cache(VCACHE& pvcache);
 
  /*
   * load edges and states from cache
   *
   * \tparam ECACHE       edges' cache type
   * \tparam VCACHE       vertices' cache type
   *
   * \param  graph_info   graph info
   * \param  pecache      edges' cache
   * \param  pvcache      factor vertices' cache
   * \param  from_factor  true  -- factor-to-variable edges | false -- variable-to-factor edges
   * \param  is_vidx      true  -- the 'idx_' field of 'bp_edata_t' records the variable's index in
   *                               the factor's adjacency list (vidx) | 
   *                      false -- records the factor's index in the variable's adjacency list (fidx)
   * 
   * \return 0 -- success, else failed
   **/
  template <typename ECACHE, typename VCACHE> 
  int load_from_cache(const graph_info_t& graph_info, ECACHE& pecache, VCACHE& pvcache, 
      bool from_factor = true, bool is_vidx = true);
 
  bool is_factor(vid_t v_i) const {
    return (local_factors_map_->get_bit(v_i) != 0);
  }

  bp_dist_size_t dist_size(vid_t v_i) const {
    return dist_sizes_.get()[v_i];
  }

  bp_dist_t dist(vid_t v_i) {
    return bp_dist_t(local_dists_buf_.get() + local_dist_offsets_.get()[v_i], dist_sizes_.get()[v_i]);
  }

  adj_unit_list_spec_t adj_list(vid_t v_i) const {
    eid_t idx_start = this->index_.get()[v_i];
    eid_t idx_end   = this->index_.get()[v_i + 1];
    return adj_unit_list_spec_t(&this->adjs_.get()[idx_start], &this->adjs_.get()[idx_end]);
  }

  bp_dist_size_t msg_offset(vid_t v_i, uint32_t idx) const {
    eid_t eidx = this->index_.get()[v_i] + idx;
    return this->adjs_.get()[eidx].edata_.msg_offset_;
  }

  /*
   * allocate a distribution state according to local vertices's distrubution size
   * 
   **/
  bp_dense_dists_t<partition_t> alloc_dist_state() const {
    return bp_dense_dists_t<partition_t>(local_dist_offsets_.get(), this->vertices_ - 1, this->partitioner_);
  }

  /*
   * allocate a message buffer according to local vertices' and its neighbours' distrubution size
   **/
  std::shared_ptr<bp_prob_t> alloc_msg_buf() const;

  static deferred_action<std::function<void()>> alloc_bsp_msg_buf() {
    // TODO:not sure return is ok
    bsp_msg_buf_p_.reset(new thread_local_buffer);
    std::function<void()> reset_func = [] { bsp_msg_buf_p_.reset(); };
    return plato::defer(std::move(reset_func));
  }

  // ******************************************************************************* //

  bp_bcsr_t(std::shared_ptr<sequence_balanced_by_source_t> partitioner, const allocator_type& alloc = ALLOC());

  bp_bcsr_t(const bp_bcsr_t&) = delete;
  bp_bcsr_t& operator=(const bp_bcsr_t&) = delete;

private:
  using traits_ = typename std::allocator_traits<ALLOC>::template rebind_traits<adj_unit_t<edata_t>>;

protected:
  using bitmap_allocator_t = typename bcsr_spec_t::bitmap_allocator_t;
  using bitmap_pointer     = typename bcsr_spec_t::bitmap_pointer;
  using adjs_allocator_t   = typename bcsr_spec_t::adjs_allocator_t;
  using adjs_pointer       = typename bcsr_spec_t::adjs_pointer;
  using index_allocator_t  = typename bcsr_spec_t::index_allocator_t;
  using index_pointer      = typename bcsr_spec_t::index_pointer;
  
  using dist_sizes_allocator_t = typename traits_::template rebind_alloc<bp_dist_size_t>;
  using dist_sizes_traits_     = typename traits_::template rebind_traits<bp_dist_size_t>;
  using dist_sizes_pointer     = typename dist_sizes_traits_::pointer;

  using dist_buf_allocator_t = typename traits_::template rebind_alloc<bp_prob_t>;
  using dist_buf_traits_     = typename traits_::template rebind_traits<bp_prob_t>;
  using dist_buf_pointer     = typename dist_buf_traits_::pointer;

  std::shared_ptr<bp_dist_size_t> dist_sizes_;

  std::shared_ptr<bitmap_spec_t>  local_factors_map_;
  std::shared_ptr<bp_dist_size_t> local_dist_offsets_;
  std::shared_ptr<bp_prob_t>      local_dists_buf_;

  bp_dist_size_t msg_buf_size_;

private:
  struct edge_msg_t {
    vid_t    v_i_;
    uint32_t idx_;
    uint32_t pos_;
  };

  struct dist_size_msg_t {
    vid_t           v_i_;
    bool            is_factor_;
    bp_dist_size_t  dist_size_;
  };

  struct dist_msg_t {
    vid_t          v_i_;
    bp_dist_size_t size_ = 0;
    bp_prob_t*     dist_ = nullptr;

    template<typename Ar>
    void serialize(Ar &ar) {
      if (!dist_) {
        dist_ = (bp_prob_t*)bsp_msg_buf_p_->local();
      }
      ar & v_i_;
      ar & size_;
      for (uint32_t i = 0; i < size_; ++i) {
        ar & dist_[i];
      }
    }
  };
};

void multiply_factor_msg(bp_dist_t& var_belief, const bp_dist_t& factor_msg) {
  // CHECK(var_belief.values_ && factor_msg.values_);
  // CHECK(var_belief.size_ == factor_msg.size_);
  // for (bp_dist_size_t i = 0; i < var_belief.size_; ++i) {
  //   var_belief.values_[i] *= factor_msg.values_[i];
  // }
  var_belief.multiply(factor_msg);
}

void multiply_variable_msg(bp_dist_t& factor_belief, const bp_dist_t& var_msg,
                          bp_dist_size_t product) {
  CHECK(factor_belief.values_ && var_msg.values_);
  CHECK(factor_belief.size_ % var_msg.size_ == 0);
  for (bp_dist_size_t i = 0; i < factor_belief.size_; ++i) {
    bp_dist_size_t idx_in_var = (i / product) % var_msg.size_;
    factor_belief.values_[i] *= var_msg.values_[idx_in_var];
  }
}

void divide_factor_msg(const bp_dist_t& var_belief, bp_dist_t& factor_msg) {
  CHECK(var_belief.values_ && factor_msg.values_);
  CHECK(var_belief.size_ == factor_msg.size_);
  for (bp_dist_size_t i = 0; i < var_belief.size_; ++i) {
    if (factor_msg.values_[i] != 0.0) {
      factor_msg.values_[i] = var_belief.values_[i] / factor_msg.values_[i];
    }
  }
}

void divide_variable_msg(const bp_dist_t& factor_belief, bp_dist_t& var_msg,
    bp_dist_size_t product) {
  CHECK(factor_belief.values_ && var_msg.values_);
  CHECK(factor_belief.size_ % var_msg.size_ == 0);
  std::vector<bp_prob_t> sum(var_msg.size_);
  for (bp_dist_size_t i = 0; i < factor_belief.size_; ++i) {
    bp_dist_size_t idx_in_var = (i / product) % var_msg.size_;
    sum[idx_in_var] += factor_belief.values_[i];
  }
  for (bp_dist_size_t i = 0; i < var_msg.size_; ++i) {
    if (var_msg.values_[i] != 0.0) {
      var_msg.values_[i] = sum[i] / var_msg.values_[i];
    }
  }
}

/*
 * run belief propagation on a bp_csr graph
 *
 * \tparam ALLOC  type of the allocator object used to define the storage allocation model
 *
 * \param graph       the bp_csr type graph
 * \param graph_info  base graph-info
 * \param opts        belief propagation options
 *
 * \return
 *      each vertex's rank value in dense representation
 **/
template <typename ALLOC>
bp_dense_dists_t<typename bp_bcsr_t<ALLOC>::partition_t> belief_propagation (
    bp_bcsr_t<ALLOC>& graph,
    const graph_info_t& graph_info,
    const bp_opts_t& opts = bp_opts_t()) {
  
  using bp_bcsr_spect_t = bp_bcsr_t<ALLOC>;
  using belief_state_t = bp_dense_dists_t<typename bp_bcsr_spect_t::partition_t>;
  using adj_unit_list_spec_t = typename bp_bcsr_spect_t::adj_unit_list_spec_t;
  using bp_msg_spec_t = typename bp_bcsr_spect_t::bp_msg_t;
  using context_spec_t = mepa_ag_context_t<bp_msg_spec_t>;
  using message_spec_t = mepa_ag_message_t<bp_msg_spec_t>;

  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  belief_state_t curt_belief = graph.alloc_dist_state();
  belief_state_t next_belief = graph.alloc_dist_state();
  auto curt_msgs_buf = graph.alloc_msg_buf();
  auto next_msgs_buf = graph.alloc_msg_buf();

  double delta = curt_belief.template foreach<double> (
    [&] (vid_t v_i, bp_dist_t* dval) {
      dval->fill(1.0);
      return 1.0 * dval->size_;
    }
  );

  auto msg_buf_reset_defer = graph.alloc_bsp_msg_buf();
  for (uint32_t epoch_i = 0; epoch_i < opts.iteration_; ++epoch_i) {
    watch.mark("t1");
    bp_prob_t* curt_msgs = curt_msgs_buf.get();
    bp_prob_t* next_msgs = next_msgs_buf.get();

    // {
    //   std::string str(">>>DEBUG: ");
    //   str += "pid:" + std::to_string(cluster_info.partition_id_) + ": "
    //       + " msgs: [";
    //   auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
    //     bool is_factor = graph.is_factor(v_i);
    //     for (auto it = adjs.begin_; it != adjs.end_; ++it) {
    //       bp_dist_t msg(&curt_msgs[it->edata_.msg_offset_], graph.dist_size(is_factor ? it->neighbour_ : v_i));
    //       str += "(" + std::to_string(v_i) + "->" + std::to_string(it->neighbour_)
    //           + ":" + msg.to_string() + ")  ";
    //     }
    //     return true;
    //   };
    //   size_t chunk_size = 1;
    //   graph.reset_traversal();
    //   while (graph.next_chunk(traversal, &chunk_size)) {}
    //   str += "]\n";
    //   std::cout << str;      
    // }

    next_belief.fill(1.0);
    plato::aggregate_message<bp_msg_spec_t, int, bp_bcsr_spect_t> (graph,
      [&](const context_spec_t& context, vid_t v_i, const adj_unit_list_spec_t& adjs) {
        bool is_factor = graph.is_factor(v_i);
        for (uint32_t idx = 0; idx < adjs.end_ - adjs.begin_; ++idx) {
          auto it = adjs.begin_ + idx;
          bp_msg_spec_t msg;
          // msg.to_ = it->neighbour_;
          msg.from_idx_ = it->edata_.idx_;
          msg.msg_.size_ = graph.dist_size(is_factor ? it->neighbour_ : v_i);
          msg.msg_.values_ = &curt_msgs[it->edata_.msg_offset_];
          context.send(message_spec_t{ it->neighbour_, msg });
        }
      },
      [&](int /*p_i*/, message_spec_t& msg) {
        vid_t v_i = msg.v_i_;
        uint32_t idx = msg.message_.from_idx_;
        bp_dist_t bp_msg = msg.message_.msg_;
        if (!graph.is_factor(v_i)) {
          multiply_factor_msg(next_belief[v_i], bp_msg);
        } else {
          bp_dist_size_t product = 1;
          auto adj_list = graph.adj_list(v_i);
          for (uint32_t i = 0; i < idx; ++i) {
            auto it = adj_list.begin_ + i;
            product *= graph.dist_size(it->neighbour_);
          }
          multiply_variable_msg(next_belief[v_i], bp_msg, product);
        }
        memcpy(next_msgs + graph.msg_offset(v_i, idx), 
              bp_msg.values_, bp_msg.size_ * sizeof(bp_prob_t));
        return 0;
      }
    );

    if (opts.iteration_ - 1 == epoch_i) {
      delta = next_belief.template foreach<double> (
        [&](vid_t v_i, bp_dist_t* dval) {
          if (graph.is_factor(v_i)) {
            dval->clean();
          } else {
            dval->multiply(graph.dist(v_i));
            dval->normalize();
          }
          return 0;
        }
      );
    } else {
      std::string str;
      delta = next_belief.template foreach<double> (
        [&] (vid_t v_i, bp_dist_t* dval) {
          dval->multiply(graph.dist(v_i));
          auto adjs = graph.adj_list(v_i);
          if (!graph.is_factor(v_i)) {
            for (uint32_t idx = 0; idx < adjs.end_ - adjs.begin_; ++idx) {
              auto it = adjs.begin_ + idx;
              bp_dist_t factor_msg(&next_msgs[it->edata_.msg_offset_],
                                graph.dist_size(v_i));
              divide_factor_msg(*dval, factor_msg);
              factor_msg.normalize();
            }
          } else {
            bp_dist_size_t product = 1;
            for (uint32_t idx = 0; idx < adjs.end_ - adjs.begin_; ++idx) {
              auto it = adjs.begin_ + idx;
              bp_dist_size_t size = graph.dist_size(it->neighbour_);
              bp_dist_t var_msg(&next_msgs[it->edata_.msg_offset_], size);
              divide_variable_msg(*dval, var_msg, product);
              var_msg.normalize();
              product *= size;
            }
          }
          dval->normalize();
          double d = 0;
          for (bp_dist_size_t i = 0; i < dval->size_; ++i) {
            d += fabs(dval->values_[i] - curt_belief[v_i].values_[i]);
          }
          return d;
        }
      );

      if (opts.eps_ > 0.0 && delta < opts.eps_) {
        epoch_i = opts.iteration_ - 2;
      }
    }

    // {
    //   std::string str(">>>DEBUG-");
    //   str += std::to_string(cluster_info.partition_id_) + ": ";
    //   str += "next_belief:[";
    //   next_belief.reset_traversal();
    //   auto trvs = [&] (vid_t v_i, bp_dist_t* dval) {
    //     str += std::to_string(v_i) + "-" + dval->to_string();
    //     return true;
    //   };
    //   size_t chunk_size = 1;
    //   while (next_belief.next_chunk(trvs, &chunk_size)) {}
    //   str += "]\n";
    //   std::cout << str;
    // }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "], delta: " << delta << ", cost: "
        << watch.show("t1") / 1000.0 << "s";
    }
    std::swap(curt_belief, next_belief);
    std::swap(curt_msgs_buf, next_msgs_buf);
  }

  return curt_belief;
}


// ************************************************************************************ //
// implementations

std::ostream& operator<<(std::ostream& os, const bp_dist_t& rhs) {
  if (!rhs.values_ || rhs.size_ == 0) {
    os << " ";
  } else {
    for (bp_dist_size_t i = 0; i < rhs.size_; ++i) {
      os << rhs.values_[i] << " ";
    }
  }
  return os;
}

/*
 * traverse the cache by bulk synchronous parallel computation model
 * 
 * \tparam CACHE        cache type
 * \tparam PART_IMPL    partitioner's type
 * \tparam DATA         data type in the cache
 * \tparam MSG          message type
 * 
 * \param  cache        the cache with DATA type objects
 * \param  trvs_task    produce task, traverse the cache and send messages, run in parallel
 * \param  recv_task    consume task, run in parallel
 * \param  chunk_size   at most process 'chunk_size' chunk at a batch
 * \param  bsp_opts     bsp options
 * \param  trvs_opts    traverse options
 * 
 * \return  0 -- success, else -- failed
 **/
template <typename DATA, typename MSG, typename CACHE>
int traverse_cache_bsp (
    CACHE& cache,
    std::function<void(DATA*, bsp_send_callback_t<MSG>)> trvs_task,
    std::function<void(bsp_recv_pmsg_t<MSG>&)> recv_task,
    size_t chunk_size,
    const bsp_opts_t bsp_opts = bsp_opts_t(),
    const traverse_opts_t& trvs_opts = traverse_opts_t()) {
  
  cache.reset_traversal(trvs_opts);
  auto __send = [&](bsp_send_callback_t<MSG> send) {
    size_t local_chunk_size = chunk_size;
    while (cache.next_chunk([&](size_t, DATA* data) {
        trvs_task(data, send);
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

/*
 * traverse the factor vertices' cache by bulk synchronous parallel computation model
 *
 **/ 
template <typename MSG, template<typename> class VCACHE>
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

  return traverse_cache_bsp<vertex_unit_t<bp_factor_data_t>, MSG>(pvacahe, traverse_task, recv_task, 1, opts);
}

template <typename PART_IMPL, typename ALLOC, typename BITMAP>
bp_dense_dists_t<PART_IMPL, ALLOC, BITMAP>::bp_dense_dists_t(
    const bp_dist_size_t* dist_offsets, vid_t max_v_i,
    std::shared_ptr<PART_IMPL> partitioner)
    : dense_state_t<bp_dist_t, PART_IMPL, ALLOC, BITMAP>(max_v_i, partitioner),
      /*buf_allocator_(),*/ dist_buf_size_(dist_offsets[max_v_i + 1]) {
    
  dist_buf_ = buf_allocator_.allocate(dist_buf_size_);
  // std::fill(dist_buf_, dist_buf_ + dist_offsets[max_v_i + 1], 1.0);

  #pragma omp parallel for
  for (vid_t i = 0; i <= max_v_i; ++i) {
    this->data_[i].size_   = dist_offsets[i + 1] - dist_offsets[i];
    this->data_[i].values_ = dist_buf_ + dist_offsets[i];
  }
}

template <typename PART_IMPL, typename ALLOC, typename BITMAP>
bp_dense_dists_t<PART_IMPL, ALLOC, BITMAP>::bp_dense_dists_t(bp_dense_dists_t&& other)
    : dense_state_t<bp_dist_t, PART_IMPL, ALLOC, BITMAP>(std::move(other)),
      buf_allocator_(other.buf_allocator_), dist_buf_size_(other.dist_buf_size_) {
  
  dist_buf_       = other.dist_buf_;
  other.dist_buf_ = nullptr;
}

template <typename PART_IMPL, typename ALLOC, typename BITMAP>
bp_dense_dists_t<PART_IMPL, ALLOC, BITMAP>& 
bp_dense_dists_t<PART_IMPL, ALLOC, BITMAP>::operator=(bp_dense_dists_t&& other) {
  if (this != &other) {
    dense_state_t<bp_dist_t, PART_IMPL, ALLOC, BITMAP>::operator=(std::move(other));
    if (nullptr != dist_buf_) {
      buf_allocator_.deallocate(dist_buf_, dist_buf_size_);
    }
    
    buf_allocator_  = other.buf_allocator_;
    dist_buf_size_  = other.dist_buf_size_;
    dist_buf_       = other.dist_buf_;
    other.dist_buf_ = nullptr;
  }
  return *this;
}

template <typename ALLOC>
std::unique_ptr<plato::thread_local_buffer> bp_bcsr_t<ALLOC>::bsp_msg_buf_p_;

template <typename ALLOC>
bp_bcsr_t<ALLOC>::bp_bcsr_t(std::shared_ptr<sequence_balanced_by_source_t> partitioner, const allocator_type& alloc)
  : bcsr_spec_t(partitioner, alloc) { }

template <typename ALLOC>
template <typename ECACHE>
int bp_bcsr_t<ALLOC>::load_from_edges_cache(const graph_info_t& graph_info, ECACHE& pecache, 
    bool from_factor, bool is_vidx) {
  CHECK(nullptr != this->partitioner_);

  vid_t tmp_vertices = 0;
  eid_t tmp_edges    = 0;

  this->vertices_  = graph_info.vertices_;
  tmp_vertices     = this->vertices_;

  {
    bitmap_allocator_t __alloc(this->allocator_);
    auto* __p = __alloc.allocate(1);
    __alloc.construct(__p, this->vertices_);

    this->bitmap_.reset(__p, [__alloc](bitmap_pointer p) mutable {
      __alloc.destroy(p);
      __alloc.deallocate(p, 1);
    });
  }

  {
    index_allocator_t __alloc(this->allocator_);
    auto* __p = __alloc.allocate(this->vertices_ + 1);
    memset(__p, 0, sizeof(eid_t) * (this->vertices_ + 1));

    this->index_.reset(__p, [__alloc, tmp_vertices](index_pointer p) mutable {
      __alloc.deallocate(p, tmp_vertices + 1);
    });
  }

  int rc = -1;
  bsp_opts_t bsp_opts;
  traverse_opts_t trvs_opts;
  trvs_opts.mode_ = traverse_mode_t::RANDOM;
  auto& cluster_info = cluster_info_t::get_instance();

  { // store vertices
    trvs_opts.auto_release_         = false;
    bsp_opts.threads_               = -1;
    bsp_opts.flying_send_per_node_  = 3;
    bsp_opts.flying_recv_           = cluster_info.partitions_;
    bsp_opts.global_size_           = 64 * MBYTES;
    bsp_opts.local_capacity_        = 32 * PAGESIZE;
    bsp_opts.batch_size_            = 1;

    auto send_srcs = [&](edge_unit_spec_t* edge, bsp_send_callback_t<vid_t> send) {
      CHECK(edge->src_ < this->vertices_);
      CHECK(edge->dst_ < this->vertices_);

      send(this->partitioner_->get_partition_id(edge->src_, edge->dst_), edge->src_);
      send(this->partitioner_->get_partition_id(edge->dst_, edge->src_), edge->dst_);
    };
    auto recv_srcs = [&](bsp_recv_pmsg_t<vid_t>& pmsg) {
      __sync_fetch_and_add(&this->index_.get()[*pmsg + 1], 1);
      this->bitmap_->set_bit(*pmsg);
    };
    if (0 != (rc = traverse_cache_bsp<edge_unit_spec_t, vid_t>(pecache, send_srcs, recv_srcs, 64, bsp_opts, trvs_opts))) {
      return rc;
    }
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-1]: count vertex's out-degree done";
  }

  // init adjs storage
  for (size_t i = 1; i <= this->vertices_; ++i) {
    this->index_.get()[i] = this->index_.get()[i] + this->index_.get()[i - 1];
  }
  this->edges_ = this->index_.get()[this->vertices_];
  tmp_edges    = this->edges_;

  LOG(INFO) << "[staging-1]: [" << cluster_info.partition_id_ << "] local edges(" << this->edges_ << ")";

  {
    mmap_allocator_t<adj_unit_spec_t> __alloc; //use mmap noreserve
    auto* __p = __alloc.allocate(this->edges_);
    if (false == std::is_trivial<adj_unit_spec_t>::value) {
      for (size_t i = 0; i < this->edges_; ++i) {
        __alloc.construct(&__p[i]);
      }
    }
    this->adjs_.reset(__p, [__alloc, tmp_edges](adjs_pointer p) mutable {
      if (false == std::is_trivial<adj_unit_spec_t>::value) {
        for (size_t i = 0; i < tmp_edges; ++i) {
          __alloc.destroy(&p[i]);
        }
      }
      __alloc.deallocate(p, tmp_edges);
    });
  }

  std::shared_ptr<eid_t> tmp_index;
  {
    index_allocator_t __alloc(this->allocator_);
    auto* __p = __alloc.allocate(this->vertices_ + 1);
    memset(__p, 0, sizeof(eid_t) * (this->vertices_ + 1));

    tmp_index.reset(__p, [__alloc, tmp_vertices](index_pointer p) mutable {
      __alloc.deallocate(p, tmp_vertices + 1);
    });
  }

  {  // store edges
    trvs_opts.auto_release_         = true;
    bsp_opts.threads_               = -1;
    bsp_opts.flying_send_per_node_  = 3;
    bsp_opts.flying_recv_           = cluster_info.partitions_;
    bsp_opts.global_size_           = 16 * MBYTES;
    bsp_opts.local_capacity_        = 32 * PAGESIZE;
    bsp_opts.batch_size_            = 1;

    auto send_edges = [&](edge_unit_spec_t* edge, bsp_send_callback_t<edge_unit_spec_t> send) {
      // only send the variable-to-factor edges with vidx or factor-to-varible edges with fidx
      if (from_factor == is_vidx) {
        auto reversed_edge = *edge;
        reversed_edge.src_ = edge->dst_;
        reversed_edge.dst_ = edge->src_;
        send(this->partitioner_->get_partition_id(edge->dst_, edge->src_), reversed_edge);
      } else {
        send(this->partitioner_->get_partition_id(edge->src_, edge->dst_), *edge);
      }
    };

    auto recv_edges = [&](bsp_recv_pmsg_t<edge_unit_spec_t>& pmsg) {
      eid_t idx = this->index_.get()[pmsg->src_] + __sync_fetch_and_add(&tmp_index.get()[pmsg->src_], (eid_t)1);

      auto& nei = this->adjs_.get()[idx];
      nei.neighbour_ = pmsg->dst_;
      nei.edata_     = pmsg->edata_;
    };
    if (0 != (rc = traverse_cache_bsp<edge_unit_spec_t, edge_unit_spec_t>(pecache, send_edges, recv_edges, 64, bsp_opts, trvs_opts))) {
      return rc;
    }
  }

  // {
  //   std::string str(">>>DEBUG: ");
  //   str += "pid:" + std::to_string(cluster_info.partition_id_) + ": "
  //       + " edges: [";
  //   auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
  //     for (eid_t i = 0; i < tmp_index.get()[v_i]; ++i) {
  //       eid_t idx = this->index_.get()[v_i] + i;
  //       auto edge = this->adjs_.get()[idx];
  //       str += "(" + std::to_string(v_i) + "," + std::to_string(edge.neighbour_)
  //           + " " + std::to_string(edge.edata_.idx_) + ") ";
  //     }
  //     return true;
  //   };
  //   size_t chunk_size = 1;
  //   this->reset_traversal();
  //   while (this->next_chunk(traversal, &chunk_size)) {}
  //   str += "]\n";
  //   std::cout << str;
  //   // exit(0);
  // }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-2]: store first part of edges done.";
  }

  // send factor-to-varibler edges with fidx or variable-to-facto edges with vidx
  aggregate_message<edge_msg_t, int, bp_bcsr_t>(*this, 
    [&](const mepa_ag_context_t<edge_msg_t>& context, vid_t v_i, const adj_unit_list_spec_t& adjs) {
      for (uint32_t idx = 0; idx < tmp_index.get()[v_i]; ++idx) {
        auto it = adjs.begin_ + idx;
        // a pair of (varible, fidx) / (factor, vidx)
        // edge_msg_t edge_pair = std::make_pair(v_i, idx);
        edge_msg_t edge_msg;
        edge_msg.v_i_ = v_i;
        edge_msg.idx_ = idx;
        edge_msg.pos_ = it->edata_.idx_;
        // send an factor-to-variable/variable-to-factor edge with fidx/vidx to factor/variable
        context.send(mepa_ag_message_t<edge_msg_t> {it->neighbour_, edge_msg});
      }
    },
    [&](int, mepa_ag_message_t<edge_msg_t>& msg) {
      auto idx = this->index_.get()[msg.v_i_] + msg.message_.pos_;
      auto& nei = this->adjs_.get()[idx];
      nei.neighbour_  = msg.message_.v_i_;   // varible/factor id
      nei.edata_.idx_ = msg.message_.idx_;   // fidx/vidx
      return 0;
    }
  );

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-3]: store second part of edges done.";
  }

  // {
  //   std::string str(">>>DEBUG: ");
  //   str += "pid:" + std::to_string(cluster_info.partition_id_) + ": "
  //       + " edges: [";
  //   auto traversal = [&](vid_t v_i, const adj_unit_list_spec_t& adjs) {
  //     for (auto it = adjs.begin_; it != adjs.end_; ++it) {
  //       str += "(" + std::to_string(v_i) + "," + std::to_string(it->neighbour_)
  //           + " " + std::to_string(it->edata_.idx_) + ") ";
  //     }
  //     return true;
  //   };
  //   size_t chunk_size = 1;
  //   this->reset_traversal();
  //   while (this->next_chunk(traversal, &chunk_size)) {}
  //   str += "]\n";
  //   std::cout << str;
  //   exit(0);
  // }

  return 0;
}

template <typename ALLOC>
template <typename VCACHE>
int bp_bcsr_t<ALLOC>::init_states_from_factors_cache(VCACHE& pvcache) {
  auto& cluster_info = cluster_info_t::get_instance();

  vid_t tmp_vertices = this->vertices_;

  {
    bitmap_allocator_t __alloc(this->allocator_);
    auto* __p = __alloc.allocate(1);
    __alloc.construct(__p, this->vertices_);

    local_factors_map_.reset(__p, [__alloc](bitmap_pointer p) mutable {
      __alloc.destroy(p);
      __alloc.deallocate(p, 1);
    });
  }

  {
    dist_sizes_allocator_t __alloc(this->allocator_);
    auto* __p = __alloc.allocate(tmp_vertices);
    memset(__p, 0, sizeof(bp_dist_size_t) * tmp_vertices);
    
    dist_sizes_.reset(__p, [__alloc, tmp_vertices](dist_sizes_pointer p) mutable {
      __alloc.deallocate(p, tmp_vertices);
    });
  }
  
  { // initialize local_factors_map_ and dist_sizes_
    bitmap_t<> local_var_map(this->vertices_);
    traverse_factors_cache_bsp<dist_size_msg_t>(pvcache,
      [&](vertex_unit_t<bp_factor_data_t>* v_data, bsp_send_callback_t<dist_size_msg_t> send) {
        auto var_size = v_data->vdata_.vars_.size();

        dist_size_msg_t factor_msg;
        factor_msg.v_i_       = v_data->vid_;
        factor_msg.is_factor_ = true;
        factor_msg.dist_size_ = var_size == 1 ? 0 : v_data->vdata_.dists_.size();
        send(this->partitioner_->get_partition_id(factor_msg.v_i_), factor_msg);

        for(const auto& var: v_data->vdata_.vars_) {
          dist_size_msg_t var_msg;
          var_msg.v_i_       = var.first;
          var_msg.is_factor_ = false;
          var_msg.dist_size_ = var.second;
          send(this->partitioner_->get_partition_id(var_msg.v_i_), var_msg);
        }
      },
      [&](bsp_recv_pmsg_t<dist_size_msg_t>& pmsg) {
        vid_t v_i = pmsg->v_i_;
        if (pmsg->is_factor_ == false) {
          CHECK(local_factors_map_->get_bit(v_i) == 0) << "vertex " << v_i << " is a factor vertex";
          if (local_var_map.get_bit(v_i)) {
            return;
          }
          local_var_map.set_bit(v_i);
        } else {
          CHECK(local_var_map.get_bit(v_i) == 0) << "vertex " << v_i << " is a variable vertex";
          CHECK(local_factors_map_->get_bit(v_i) == 0) << "duplicated factor vertex: " << v_i;
          local_factors_map_->set_bit(v_i);
        }
        dist_sizes_.get()[v_i] = pmsg->dist_size_;
      });
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-4]: count vertice's distribution size done.";
  } 

  {
    dist_sizes_allocator_t __alloc(this->allocator_);
    auto* __p = __alloc.allocate(tmp_vertices + 1);
    memset(__p, 0, sizeof(bp_dist_size_t) * (tmp_vertices + 1));

    local_dist_offsets_.reset(__p, [__alloc, tmp_vertices](dist_sizes_pointer p) mutable {
      __alloc.deallocate(p, tmp_vertices + 1);
    });
  }

  #pragma omp parallel for
  for (vid_t v_i = 1; v_i <= this->vertices_; ++v_i) {
    local_dist_offsets_.get()[v_i] = dist_sizes_.get()[v_i - 1];
  }

  for (vid_t v_i = 1; v_i <= this->vertices_; ++v_i) {
    local_dist_offsets_.get()[v_i] += local_dist_offsets_.get()[v_i - 1];
  }

  // sync dist_sizes_ between all partitions after buiding local_dist_offsets_
  allreduce(MPI_IN_PLACE, dist_sizes_.get(), this->vertices_, get_mpi_data_type<bp_dist_size_t>(), MPI_SUM, MPI_COMM_WORLD);

  // {
  //   std::string str(">>>DEBUG-");
  //   str += std::to_string(cluster_info.partition_id_) + ": ";
  //   str += "dist_sizes: [";
  //   for (vid_t i = 0; i < this->vertices_; ++i) {
  //     str += "(" + std::to_string(i) + "-" + std::to_string(dist_sizes_.get()[i]) + ") ";
  //   }
  //   str += "]\n";
  //   str += "dist_offsets: [";
  //   for (vid_t i = 0; i <= this->vertices_; ++i) {
  //     str += std::to_string(local_dist_offsets_.get()[i]) + " ";
  //   }
  //   str += "]\n";
  //   std::cout << str;
  //   exit(0);
  // }

  {
    dist_buf_allocator_t __alloc(this->allocator_);
    bp_dist_size_t __local_dist_size = local_dist_offsets_.get()[this->vertices_];
    auto* __p = __alloc.allocate(__local_dist_size);
    // initialize the probability distribution to all 1.0
    std::fill(__p, __p + __local_dist_size, bp_prob_t(1.0));

    local_dists_buf_.reset(__p, [__alloc, __local_dist_size](dist_buf_pointer p) mutable {
      __alloc.deallocate(p, __local_dist_size);
    });
  }

  { // initialize values of local_dists_buf_
    bsp_msg_buf_p_.reset(new thread_local_buffer);
    auto msg_buf_reset_defer = plato::defer([] { bsp_msg_buf_p_.reset(); });
    traverse_factors_cache_bsp<dist_msg_t>(pvcache, 
      [&] (plato::vertex_unit_t<bp_factor_data_t>* v_data, plato::bsp_send_callback_t<dist_msg_t> send) {
        dist_msg_t msg;
        msg.size_ = v_data->vdata_.dists_.size();
        msg.dist_ = v_data->vdata_.dists_.data();
        if (v_data->vdata_.vars_.size() > 1) {
          msg.v_i_ = v_data->vid_;
        } else {
          // use the variable to replace the factor
          msg.v_i_ = v_data->vdata_.vars_.front().first;
        }
        send(this->partitioner_->get_partition_id(msg.v_i_), msg);
      },
      [&](plato::bsp_recv_pmsg_t<dist_msg_t>& pmsg) {
        vid_t v_i = pmsg->v_i_;
        CHECK(pmsg->size_ == local_dist_offsets_.get()[v_i + 1] - local_dist_offsets_.get()[v_i]);
        memcpy(local_dists_buf_.get() + local_dist_offsets_.get()[v_i], pmsg->dist_, sizeof(bp_prob_t) * pmsg->size_);
      });
  }

  // {
  //   std::string str(">>>DEBUG-");
  //   str += std::to_string(cluster_info.partition_id_) + ": ";
  //   str += "dists: [";
  //   for (vid_t i = 0; i < this->vertices_; ++i) {
  //     auto start = local_dist_offsets_.get()[i], end = local_dist_offsets_.get()[i + 1];
  //     if (end - start != 0) {
  //       str += std::to_string(i) + "-(";
  //       for (auto idx = start; idx < end; ++idx) {
  //         str += std::to_string(local_dists_buf_.get()[idx]) + " ";
  //       }
  //       str += ") ";
  //     }
  //   }
  //   str += "]\n";
  //   std::cout << str;
  //   exit(0); 
  // }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-5]: store distributions done.";
  } 

  std::shared_ptr<bp_dist_size_t> local_msg_offsets;
  {
    dist_sizes_allocator_t __alloc(this->allocator_);
    auto* __p = __alloc.allocate(tmp_vertices + 1);
    memset(__p, 0, sizeof(bp_dist_size_t) * (tmp_vertices + 1));

    local_msg_offsets.reset(__p, [__alloc, tmp_vertices](dist_sizes_pointer p) mutable {
      __alloc.deallocate(p, tmp_vertices + 1);
    });
  }

  #pragma omp parallel for
  for (vid_t v_i = 0; v_i < this->vertices_; ++v_i) {
    if (this->bitmap_->get_bit(v_i) == 0) {
      continue;
    }
    bp_dist_size_t msg_size = 0;
    // compute the total buffer size of the vertex's messages to send
    if (local_factors_map_->get_bit(v_i)) {
        for (eid_t idx = this->index_.get()[v_i]; idx < this->index_.get()[v_i + 1]; ++idx) {
          vid_t vid = this->adjs_.get()[idx].neighbour_;
          msg_size += dist_sizes_.get()[vid];
        }
    } else {
      msg_size += dist_sizes_.get()[v_i] * (this->index_.get()[v_i + 1] - this->index_.get()[v_i]);
    }
    local_msg_offsets.get()[v_i + 1] = msg_size;
  }

  for (vid_t v_i = 1; v_i <= this->vertices_; ++v_i) {
    local_msg_offsets.get()[v_i] += local_msg_offsets.get()[v_i - 1];
  }
  msg_buf_size_ = local_msg_offsets.get()[this->vertices_];

  // initialize the msg_offset_ in edata_ of each edge
  #pragma omp parallel for
  for (vid_t v_i = 0; v_i < this->vertices_; ++v_i) {
    if (this->bitmap_->get_bit(v_i) == 0) {
      continue;
    }
    bp_dist_size_t offset = local_msg_offsets.get()[v_i];
    if (local_factors_map_->get_bit(v_i)) {
      for (eid_t idx = this->index_.get()[v_i]; idx < this->index_.get()[v_i + 1]; ++idx) {
        auto& edge = this->adjs_.get()[idx];
        edge.edata_.msg_offset_ = offset;
        offset += dist_sizes_.get()[edge.neighbour_];
      }
    } else {
      for (eid_t idx = this->index_.get()[v_i]; idx < this->index_.get()[v_i + 1]; ++idx) {
        auto& edge = this->adjs_.get()[idx];
        edge.edata_.msg_offset_ = offset;
        offset += dist_sizes_.get()[v_i];
      }
    }
  }

  // {
  //   std::string str(">>>DEBUG-");
  //   str += std::to_string(cluster_info.partition_id_) + ": ";
  //   str += "msg_offsets: [";
  //   for (vid_t v_i = 0; v_i < this->vertices_; ++v_i) {
  //     if (this->bitmap_->get_bit(v_i) == 0) {
  //       continue;
  //     }
  //     for (eid_t idx = this->index_.get()[v_i]; idx < this->index_.get()[v_i + 1]; ++idx) {
  //       auto& edge = this->adjs_.get()[idx];
  //       str += "(" + std::to_string(v_i) + "," + std::to_string(edge.neighbour_)
  //           + " " + std::to_string(edge.edata_.msg_offset_)  + ") ";
  //     }
  //   }
  //   str += "]\n";
  //   str += "total_msg_buf: " + std::to_string(msg_buf_size_) + "\n";
  //   std::cout << str;
  //   exit(0);
  // }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "[staging-6]: initialize the message offset of each edge done.";
  } 

  return 0;
}

template <typename ALLOC>
template <typename ECACHE, typename VCACHE> 
int bp_bcsr_t<ALLOC>::load_from_cache(const graph_info_t& graph_info, ECACHE& pecache, VCACHE& pvcache, 
    bool from_factor, bool is_vidx) {
  int rc = -1;
  if (0 != (rc = load_from_edges_cache(graph_info, pecache, from_factor, is_vidx))) {
    return rc;
  }
  return init_states_from_factors_cache(pvcache);
}

template <typename ALLOC>
std::shared_ptr<bp_prob_t> bp_bcsr_t<ALLOC>::alloc_msg_buf() const {
  dist_buf_allocator_t __alloc(this->allocator_);
  bp_dist_size_t __msg_buf_size = msg_buf_size_;
  auto* __p = __alloc.allocate(__msg_buf_size);
  // initialize the probability distribution to all 1.0
  std::fill(__p, __p + __msg_buf_size, bp_prob_t(1.0));

  std::shared_ptr<bp_prob_t> msg_buf(__p, [__alloc, __msg_buf_size](dist_buf_pointer p) mutable {
    __alloc.deallocate(p, __msg_buf_size);
  });
  return msg_buf;
}

// ************************************************************************************ //

}}  // namespace algo, namespace plato

#endif