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

using bp_prob_t        = plato::algo::bp_prob_t;
using bp_dist_size_t   = plato::algo::bp_dist_size_t;
using bp_factor_data_t = plato::algo::bp_factor_data_t;
using bp_edata_t       = plato::algo::bp_edata_t;
using bp_bcsr_t        = plato::algo::bp_bcsr_t<>;

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
      bp_dist_size_t total_states = 1;
      while(pVar) {
        char* pDash = std::strchr(pVar, '-');
        if (pDash) {
          *pDash = '\0';
          plato::vid_t vid = std::strtoul(pVar, nullptr, 0);
          bp_dist_size_t state = std::strtoul(pDash + 1, nullptr, 0);
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
          bp_prob_t dist = std::strtod(pDash + 1, nullptr);
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

  auto traversal = [&](size_t, plato::vertex_unit_t<bp_factor_data_t>* v_data) {
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
  };
  
  pvcache.reset_traversal();
  #pragma omp parallel
  {
    size_t chunk_size = 1;
    while (pvcache.next_chunk(traversal, &chunk_size)) {}
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
  CHECK(0 == pbcsr->load_from_cache(*pgraph_info, *pecache, *pvcache));

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "build bp_bcsr cost: " << watch.show("t1") / 1000.0 << "s";
    LOG(INFO) << "total cost:         " << watch.show("t0") / 1000.0 << "s";
  }

  return pbcsr;
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