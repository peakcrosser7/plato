#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "plato/graph/graph.hpp"
#include "plato/algo/bp/belief_propagation.hpp"

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

using bp_state_t       = plato::algo::bp_state_t;
using bp_dist_t        = plato::algo::bp_dist_t;
using bp_factor_data_t = plato::algo::bp_factor_data_t;
using bp_bcsr_t        = plato::algo::bp_bcsr_t;
using bp_edata_t          = plato::algo::bp_edata_t;

template <template<typename> class VCACHE>
std::shared_ptr<VCACHE<bp_factor_data_t>> load_factors_cache(const std::string& path) {

  auto pvcache = plato::load_vertices_cache<bp_factor_data_t, VCACHE>(
      path, plato::edge_format_t::CSV, [&](bp_factor_data_t* item, char* content) {
      char* pDists = nullptr;
      char* pVars = nullptr;
      pVars = strtok_r(content, ",", &pDists);
      if (!pVars || !pDists) {
          return false;
      }

      char* pSave = nullptr;
      char* pVar = nullptr;
      pVar = strtok_r(pVars, " ", &pSave);
      bp_state_t total_states = 1;
      while(pVar) {
          char* pDash = std::strchr(pVar, '-');
          if (pDash) {
          *pDash = '\0';
          plato::vid_t vid = std::strtoul(pVar, nullptr, 0);
          bp_state_t state = std::strtoul(pDash + 1, nullptr, 0);
          item->vars_.emplace_back(vid, state);
          total_states *= state;
          }
          pVar = strtok_r(nullptr, " ", &pSave);
      }

      item->dists_.resize(total_states);
      char* pDist = nullptr;
      pDist = strtok_r(pDists, " ", &pSave);
      while(pDist) {
          char* pDash = std::strchr(pDist, '-');
          if (pDash) {
          *pDash = '\0';
          auto index = std::strtoul(pDist, nullptr, 0);
          bp_dist_t dist = std::strtod(pDash + 1, nullptr);
          item->dists_[index] = dist;
          }
          pDist = strtok_r(nullptr, " ", &pSave);
      }
      return true;
      });

  return pvcache;
}

template <template<typename, typename> class ECACHE, template<typename> class VCACHE>
std::shared_ptr<ECACHE<bp_edata_t, plato::vid_t>> load_edges_cache_from_factors_cache(
    VCACHE<bp_factor_data_t>& pvcache, plato::graph_info_t* pginfo) {

  plato::eid_t edges = 0;
  plato::bitmap_t<> v_bitmap(std::numeric_limits<vid_t>::max());
  std::shared_ptr<ECACHE<bp_edata_t, plato::vid_t>> pecache(new ECACHE());

  pvcache.reset_traversal();
  #pragma omp parallel
  {
      size_t chunk_size = 1;
      while (pvcache.next_chunk([&](size_t, plato::vertex_unit_t<bp_factor_data_t>* v_data) {
      plato::eid_t n_vars = v_data->vdata_.vars_.size();
      v_bitmap.set_bit(v_data->vid_);
      // use variable vertex to replace factor vertex
      if (n_vars == 1) {
          v_bitmap.set_bit(v_data->vdata_.vars_.front().first);
          return true;
      }
      __sync_fetch_and_add(&edges, n_vars);
      for(uint32_t i = 0; i < n_vars; ++i) {
          const auto& var = v_data->vdata_.vars_[i];
          v_bitmap.set_bit(var.first);
          plato::edge_unit_t<bp_edata_t, plato::vid_t> edge;
          edge.src_ = v_data->vid_;
          edge.dst_ = var.first;
          // the var's index in factor's adjs
          edge.edata_.idx_ = i;
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

template <template<typename, typename> class ECACHE = plato::edge_block_cache_t, template<typename> class VCACHE = plato::vertex_cache_t>
std::shared_ptr<bp_bcsr_t> create_bp_bcsr_seq_from_path(
    plato::graph_info_t*  pgraph_info,
    const std::string&    path,
    int                   alpha = -1,
    bool                  use_in_degree = false) {

  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  watch.mark("t0");

  auto pvcache = load_factors_cache<VCACHE>(path);

  watch.mark("t1");

  auto pecache = load_edges_cache_from_factors_cache<ECACHE>(pvcache, pgraph_info);
  
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "edges:        " << pgraph_info->edges_;
    LOG(INFO) << "vertices:     " << pgraph_info->vertices_;
    LOG(INFO) << "max_v_id:     " << pgraph_info->max_v_i_;
    LOG(INFO) << "is_directed_: " << pgraph_info->is_directed_;

    LOG(INFO) << "load edges cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<partition_t> part_bcsr = nullptr;
  {
    std::vector<plato::vid_t> degrees;
    if (use_in_degree) {
      degrees = generate_dense_in_degrees<plato::vid_t>(*pgraph_info, *pecache);
    } else {
      degrees = generate_dense_out_degrees<plato::vid_t>(*pgraph_info, *pecache);
    }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "generate degrees cost: " << watch.show("t1") / 1000.0 << "s";
    }
    watch.mark("t1");

    plato::eid_t __edges = pgraph_info->edges_;
    // factor graph is a undirected
    __edges = __edges * 2;

    part_bcsr.reset(new partition_t(degrees.data(), pgraph_info->vertices_,
      __edges, alpha));
    part_bcsr->check_consistency();
  }

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "partition cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  std::shared_ptr<bp_bcsr_t> pbcsr(new bp_bcsr_t(part_bcsr));
  CHECK(0 == pbcsr->load_from_ecache(*pgraph_info, *pecache));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build dcsc cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "total cost:      " << watch.show("t0") / 1000.0 << "s";
  }

  return pbcsr;
}












/*
static std::shared_ptr<bp_state_t> vars_dist_mmap_p;
static std::shared_ptr<bp_dist_t> dist_mmap_p;
static std::shared_ptr<bp_dist_t> bp_mmap_p;
static std::shared_ptr<bp_dist_t*> msg_ptr_mmap_p;
static std::shared_ptr<bp_dist_t> msg_mmap_p;

template <typename VCACHE, typename GRAPH>
void init_states_from_factors_cache(
    VCACHE& pvcache,
    GRAPH& graph,
    const plato::graph_info_t& graph_info) {

  auto partitioner = graph.partitioner();

  plato::bitmap_t<> local_factor_map(graph_info.vertices_);
  bp_state_t local_dist_size = 0;
  
  {
    plato::bitmap_t<> local_var_map(graph_info.vertices_);
    traverse_factors_cache(pvcache, graph_info,
      [&](plato::vertex_unit_t<bp_factor_data_t>* v_data, plato::bsp_send_callback_t<bp_dist_size_msg_t> send) {
        auto var_size = v_data->vdata_.vars_.size();
        if (var_size > 1) {
          bp_dist_size_msg_t factor_msg;
          factor_msg.vid_       = v_data->vid_;
          factor_msg.v_type_    = bp_vertex_type_t::FACTOR;
          factor_msg.dist_size_ = v_data->vdata_.dists_.size();
          send(partitioner->get_partition_id(factor_msg.vid_), factor_msg);
        }
        for(const auto& var: v_data->vdata_.vars_) {
          bp_dist_size_msg_t var_msg;
          var_msg.vid_       = var.first;
          var_msg.v_type_    = bp_vertex_type_t::VARIABLE;
          var_msg.dist_size_ = var.second;
          send(partitioner->get_partition_id(var_msg.vid_), var_msg);
        }
      },
      [&](plato::bsp_recv_pmsg_t<bp_dist_size_msg_t>& pmsg) {
        plato::vid_t vid = pmsg->vid_;
        if (pmsg->v_type_ == bp_vertex_type_t::VARIABLE) {
          CHECK(local_factor_map.get_bit(vid) == 0) << "vertex " << vid << " is a factor vertex";
          if (local_var_map.get_bit(vid) != 0) {
            return;
          }
          local_var_map.set_bit(vid);
        } else {
          CHECK(local_var_map.get_bit(vid) == 0) << "vertex " << vid << " is a variable vertex";
          CHECK(local_factor_map.get_bit(vid) == 0) << "duplicated factor vertex: " << vid;
          local_factor_map.set_bit(vid);
        }
        __sync_fetch_and_add(&local_dist_size, pmsg->dist_size_);
      });
  }

  uint64_t local_factor_edges = 0;

  auto odegrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, graph, false);
  odegrees.template foreach<int>(
    [&](plato::vid_t v_i, uint32_t* pval) {
      __sync_fetch_and_add(&local_factor_edges, *pval);
      return 0;
    }, local_factor_map);

  vars_dist_mmap_p.reset((bp_state_t*)
    mmap(nullptr, sizeof(bp_state_t) * local_factor_edges, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    plato::mmap_deleter{sizeof(bp_state_t) * local_factor_edges});
  CHECK(MAP_FAILED != vars_dist_mmap_p.get())
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno) << " local_factor_edges: " << local_factor_edges;
  
  plato::sparse_state_t<bp_state_t*, plato::empty_t> factor_states(local_factor_map.count(), std::make_shared<plato::empty_t>());

  {
    msg_buffer_p.reset(new plato::thread_local_buffer);
    auto adjs_buffer_reset_defer = plato  ::defer([] { msg_buffer_p.reset(); });

    traverse_factors_cache(pvcache, graph_info,
      [&](plato::vertex_unit_t<bp_factor_data_t>* v_data, plato::bsp_send_callback_t<bp_var_dist_msg_t> send) {
        const auto& vars = v_data->vdata_.vars_;
        if (vars.size() > 1) {
          bp_var_dist_msg_t msg;
          msg.num_vars_ = vars.size();
          msg.var_dist_ = vars.data();
        }
      });
  }
  
  
  dist_mmap_p.reset((bp_dist_t*)
    mmap(nullptr, sizeof(bp_dist_t) * local_dist_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    plato::mmap_deleter{sizeof(bp_dist_t) * local_dist_size});
  CHECK(MAP_FAILED != vars_dist_mmap_p.get())
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno) << " local_dist_size: " << local_dist_size;
  bp_mmap_p.reset((bp_dist_t*)
    mmap(nullptr, sizeof(bp_dist_t) * local_dist_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    plato::mmap_deleter{sizeof(bp_dist_t) * local_dist_size});
  CHECK(MAP_FAILED != bp_mmap_p.get())
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno) << " local_dist_size: " << local_dist_size;
  msg_ptr_mmap_p.reset((bp_dist_t**)
    mmap(nullptr, sizeof(bp_dist_t*) * graph.edges(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0),
    plato::mmap_deleter{sizeof(bp_dist_t*) * graph.edges()});
  CHECK(MAP_FAILED != vars_dist_mmap_p.get())
    << boost::format("WARNING: mmap failed, err code: %d, err msg: %s") % errno % strerror(errno) << " local_dist_size: " << local_dist_size;

  plato::dense_state_t<bp_vertex_state_t, GRAPH::partition_t> states(graph_info.max_v_i_, partitioner);




  // TODO:
  // 1.alloc factors' var_states arr
  // 2.update factor_var_states and factor_map by bsp and pvcache


  // TODO:
  // 1. compute total out_degree
  // 2. alloc 
  // 3. alloc all vertices' dist arr and bp arr
  // 4. init all vertices' states -- dist
  // 5. alloc all vertices' msgs_ptrs arr and msgs arr

}
*/


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