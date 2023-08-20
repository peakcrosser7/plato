#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/util/perf.hpp"
#include "plato/util/hdfs.hpp"
#include "plato/util/atomic.hpp"
#include "plato/graph/base.hpp"
#include "plato/graph/state.hpp"
#include "plato/graph/structure.hpp"
#include "plato/graph/message_passing.hpp"

DEFINE_string(input_factors, "", "input factor vertices file, in csv format");
DEFINE_string(output,      "",      "output directory");
DEFINE_bool(part_by_in,    false,   "partition by in-degree");
DEFINE_int32(alpha,        -1,      "alpha value used in sequence balance partition");
DEFINE_uint64(iterations,  100,     "number of iterations");

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input_factors,  &string_not_empty);
DEFINE_validator(output,         &string_not_empty);

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

using state_t = uint64_t;

template <typename T>
struct factor_vertex_t {
  std::vector<std::pair<plato::vid_t, state_t>> vars;
  std::vector<T> dists;
};

template <typename VCACHE>
std::shared_ptr<VCACHE> load_factors_cache(const std::string& path) {

  using dist_t = double;
  using vdata_t = factor_vertex_t<dist_t>;
  
  auto pvcache = plato::load_vertices_cache<vdata_t>(
    path, plato::edge_format_t::CSV, [&](vdata_t* item, char* content) {
      char* pDists = nullptr;
      char* pVars = nullptr;
      pVars = strtok_r(content, ",", &pDists);
      if (!pVars || !pDists) {
        return false;
      }

      char* pSave = nullptr;
      char* pVar = nullptr;
      pVar = strtok_r(pVars, " ", &pSave);
      state_t total_states = 1;
      while(pVar) {
        char* pDash = std::strchr(pVar, '-');
        if (pDash) {
          *pDash = '\0';
          plato::vid_t vid = std::strtoul(pVar, nullptr, 0);
          state_t state = std::strtoul(pDash + 1, nullptr, 0);
          item->vars.emplace_back(vid, state);
          total_states *= state;
        }
        pVar = strtok_r(nullptr, " ", &pSave);
      }

      item->dists.resize(total_states);
      char* pDist = nullptr;
      pDist = strtok_r(pDists, " ", &pSave);
      while(pDist) {
        char* pDash = std::strchr(pDist, '-');
        if (pDash) {
          *pDash = '\0';
          auto index = std::strtoul(pDist, nullptr, 0);
          dist_t dist = std::strtod(pDash + 1, nullptr);
          item->dists[index] = dist;
        }
        pDist = strtok_r(nullptr, " ", &pSave);
      }
      return true;
    });
  
  return pvcache;
}

template <typename ECACHE, typename VCACHE>
std::shared_ptr<ECACHE> load_edges_cache_from_factors_cache(
    VCACHE& pvcache, plato::graph_info_t* pginfo) {

  plato::eid_t edges = 0;
  plato::bitmap_t<> v_bitmap(std::numeric_limits<vid_t>::max());
  std::shared_ptr<ECACHE> pecache(new ECACHE());

  pvcache.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (pvcache.next_chunk([&](size_t, plato::vertex_unit_t<factor_vertex_t<double>>* v_data) {
      plato::eid_t var_size = v_data->vdata_.vars.size();
      v_bitmap.set_bit(v_data->vid_);
      // use variable vertex to replace factor vertex
      if (var_size == 1) {
        v_bitmap.set_bit(v_data->vdata_.vars.front().first);
        return true;
      }
      __sync_fetch_and_add(&edges, var_size);
      for(const auto& var: v_data->vdata_.vars) {
        v_bitmap.set_bit(var.first);
        plato::edge_unit_t<edata_t, plato::vid_t> edge;
        edge.src_ = v_data->vid_;
        edge.dst_ = v ar.first;
        pecache->push_back(edge);
      }
      return true;
    })) {}
  }

  MPI_Allreduce(MPI_IN_PLACE, &edges, 1, plato::get_mpi_data_type<plato::eid_t>(), MPI_SUM, MPI_COMM_WORLD);
  v_bitmap.sync();
  
  if (pginfo) {
    pginfo->edges_    = edges;
    pginfo->vertices_ = v_bitmap.count();
    pginfo->max_v_i_  = v_bitmap.msb();
  }

  return pecache;
}

template <typename SEQ_PART, template<typename, typename> class CACHE = plato::edge_block_cache_t, typename VCACHE>
std::shared_ptr<plato::dcsc_t<plato::empty_t, SEQ_PART>> create_dcsc_seq_from_factors(
    VCACHE&               pvcache,
    plato::graph_info_t*  pgraph_info,
    int                   alpha = -1,
    bool                  use_in_degree = false) {

  static_assert(std::is_same<SEQ_PART, plato::sequence_balanced_by_source_t>::value
      || std::is_same<SEQ_PART, plato::sequence_balanced_by_destination_t>::value, "invalid SEQ_PART type!");

  using edata_t = plato::empty_t;
  using dcsc_spec_t = plato::dcsc_t<edata_t, SEQ_PART>;

  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  watch.mark("t0");
  watch.mark("t1");

  auto cache = load_edges_cache_from_factors_cache<CACHE<edata_t, plato::vid_t>>(
    pvcache, pgraph_info);
  
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << pgraph_info->edges_;
    LOG(INFO) << "vertices:     " << pgraph_info->vertices_;
    LOG(INFO) << "max_v_id:     " << pgraph_info->max_v_i_;
    LOG(INFO) << "is_directed_: " << pgraph_info->is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<SEQ_PART> part_dcsc = nullptr;
  {
    std::vector<plato::vid_t> degrees;
    if (use_in_degree) {
      degrees = generate_dense_in_degrees<plato::vid_t>(*pgraph_info, *cache);
    } else {
      degrees = generate_dense_out_degrees<plato::vid_t>(*pgraph_info, *cache);
    }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "generate degrees cost: " << watch.show("t1") / 1000.0 << "s";
    }
    watch.mark("t1");

    plato::eid_t __edges = pgraph_info->edges_;
    if (false == pgraph_info->is_directed_) { __edges = __edges * 2; }

    part_dcsc.reset(new SEQ_PART(degrees.data(), pgraph_info->vertices_,
      __edges, alpha));
    part_dcsc->check_consistency();
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<dcsc_spec_t> pdcsc(new dcsc_spec_t(part_dcsc));
  CHECK(0 == pdcsc->load_from_cache(*pgraph_info, *cache));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build dcsc cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "total cost:      " << watch.show("t0") / 1000.0 << "s";
  }

  return pdcsc;
}

enum class bp_vertex_type_t {
  VARIABLE = 0,
  FACTOR = 1
};

struct bp_state_msg_t {
  plato::vid_t vid;
  bp_vertex_type_t v_type;
  state_t dist_size;
};

template <typename VCACHE, typename PART_IMPL>
state_t generate_local_dist_size(
    VCACHE& pvacahe,
    const plato::graph_info_t& graph_info,
    std::shared_ptr<PART_IMPL> partitioner) {

  plato::bitmap_t<> visited(graph_info.vertices_);

  auto& cluster_info = cluster_info_t::get_instance();

  state_t local_dist_size = 0;

  plato::bsp_opts_t opts;
  opts.threads_               = -1;
  opts.flying_send_per_node_  = 3;
  opts.flying_recv_           = cluster_info.partitions_;
  opts.global_size_           = 16 * MBYTES;
  opts.local_capacity_        = PAGESIZE;
  opts.batch_size_            = 1;

  pvcache.reset_traversal();
  auto __send = [&](plato::bsp_send_callback_t<bp_state_msg_t> send) {
    size_t chunk_size = 1;
    while (pvcache.next_chunk([&](size_t, plato::vertex_unit_t<factor_vertex_t<double>>* v_data) {
        plato::eid_t var_size = v_data->vdata_.vars.size();
        if (var_size != 1) {
          bp_state_msg_t factor_state;
          factor_state.vid       = v_data->vid_;
          factor_state.v_type    = bp_vertex_type_t::FACTOR;
          factor_state.dist_size = v_data->vdata_.dists.size();
          send(partitioner->get_partition_id(factor_state.vid), factor_state);
        }
        for(const auto& var: v_data->vdata_.vars) {
          bp_state_msg_t var_state;
          var_state.vid          = var.first;
          var_state.v_type       = bp_vertex_type_t::VARIABLE;
          factor_state.dist_size = var.second;
          send(partitioner->get_partition_id(var_state.vid), var_state);
        }
        return true;
      }, &chunk_size)) {}
  };

  auto __recv = [&] (int /* p_i */, plato::bsp_recv_pmsg_t<bp_state_msg_t>& pmsg) {
    if (visited.get_bit(pmsg->vid) == 0) {
      visited.set_bit(pmsg->vid);
      __sync_fetch_and_add(&local_dist_size, pmsg->dist_size);
    }
  };

  auto rc = fine_grain_bsp<bp_state_msg_t>(__send, __recv, opts);
  if (0 != rc) {
    LOG(ERROR) << "bsp failed with code: " << rc;
    return -1;
  }

  return local_dist_size;
}

template <typename VCACHE, typename GRAPH>
void init_states_from_factors_cache(
    VCACHE& pvcache,
    GRAPH& graph,
    const plato::graph_info_t& graph_info) {
    
  auto partitioner = graph.partitioner();

  state_t local_dist_size = generate_local_dist_size(pvcache, graph_info, partitioner);
  
  // TODO:
  // 1. compute total out_degree
  // 2. alloc factors' var_states arr,
  // 3. alloc all vertices' dist arr and bp arr
  // 4. init all vertices' states -- dist
  // 5. alloc all vertices' msgs_ptrs arr and msgs arr

}

int main(int argc, char** argv) {
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  watch.mark("t0");

  // init graph
  plato::graph_info_t graph_info;

  return 0;
}