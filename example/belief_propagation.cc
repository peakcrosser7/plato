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

DEFINE_string(input_factors,  "",      "input factor vertices file, in csv format");
DEFINE_string(output,         "",      "output directory");
DEFINE_bool(part_by_in,       false,   "partition by in-degree");
DEFINE_int32(alpha,           -1,      "alpha value used in sequence balance partition");
DEFINE_uint64(iterations,     100,     "number of iterations");

DEFINE_double(
  eps,
  0.001,
  "the calculation will be considered complete if the sum of "
  "the difference of the 'belief' value between iterations changes "
  "less than 'eps'. if 'eps' equals to 0, belief propagation will "
  "be force to execute 'iteration' epochs."
);

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

using bp_dist_size_t   = plato::algo::bp_dist_size_t;
using bp_prob_t        = plato::algo::bp_prob_t;
using bp_dist_t        = plato::algo::bp_dist_t;
using bp_factor_data_t = plato::algo::bp_factor_data_t;
using bp_edata_t       = plato::algo::bp_edata_t;
using bp_bcsr_t        = plato::algo::bp_bcsr_t<>;

/*
 * parallel load factor vertex data from file system to cache
 *
 * \tparam VCACHE       vertices' cache type, can be 'vertex_cache_t' or 'vertex_file_cache_t'
 *
 * \param  path         input file path, 'path' can be a file or a directory.
 *                      'path' can be located on hdfs or posix, distinguish by its prefix.
 *                      eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param  format       file format
 *
 * \return loaded factors cache or nullptr
 **/
template <template<typename> class VCACHE>
std::shared_ptr<VCACHE<bp_factor_data_t>> load_factors_cache(
    const std::string& path, plato::edge_format_t format) {

  auto pvcache = plato::load_vertices_cache<bp_factor_data_t, VCACHE>(
    path, format, [&](bp_factor_data_t* item, char* content) {
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

/*
 * parallel load edges to cache from factors cache
 *
 * \tparam ECACHE    edges' cache type, can be 'edge_block_cache_t' or 'edge_file_cache_t' or
 *                   'edge_cache_t'
 * \tparam VCACHE    vertices' cache type
 *
 * \param  pvcahe     factor vertices' cache
 * \param  pginfo     graph info
 *
 * \return loaded edges cache or nullptr
 **/
template <template<typename, typename> class ECACHE, template<typename> class VCACHE>
std::shared_ptr<ECACHE<bp_edata_t, plato::vid_t>> load_edges_cache_from_factors_cache(
    plato::graph_info_t* pginfo, VCACHE<bp_factor_data_t>& pvcache) {

  plato::eid_t edges = 0;
  plato::bitmap_t<> v_bitmap(std::numeric_limits<plato::vid_t>::max());
  std::shared_ptr<ECACHE<bp_edata_t, plato::vid_t>> pecache(new ECACHE<bp_edata_t, plato::vid_t>());

  auto traversal = [&](size_t, plato::vertex_unit_t<bp_factor_data_t>* v_data) {
    plato::eid_t n_vars = v_data->vdata_.vars_.size();
    v_bitmap.set_bit(v_data->vid_);
    // use variable vertex to replace factor vertex 
    // when the factor vertex conect only one variable vertex
    if (n_vars == 1) {
      v_bitmap.set_bit(v_data->vdata_.vars_.front().first);
      return true;
    }
    __sync_fetch_and_add(&edges, n_vars);
    for(uint32_t i = 0; i < n_vars; ++i) {
      const auto& var_pair = v_data->vdata_.vars_[i];
      v_bitmap.set_bit(var_pair.first);
      // an factor-to-variable edge
      plato::edge_unit_t<bp_edata_t, plato::vid_t> edge;
      edge.src_ = v_data->vid_;
      edge.dst_ = var_pair.first;
      // the variable's index in factor's adjs
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
    pginfo->is_directed_ = false;
    pginfo->edges_       = edges;
    pginfo->vertices_    = v_bitmap.count();
    pginfo->max_v_i_     = v_bitmap.msb();
  }

  return pecache;
}

/*
 * create bp_bcsr graph structure with sequence balanced by source partition from file system
 *
 * \tparam ECACHE         edges' cache type, can be 'edge_block_cache_t' or 'edge_file_cache_t'
 *                        or 'edge_cache_t'
 * \tparam VCACHE         vertices' cache type, can be 'vertex_cache_t' or 'vertex_file_cache_t'
 * 
 * \param pgraph_info     this function will fill all fields of graph_info_t during load process.
 * \param path            input factor vertices file path, 'path' can be a file or a directory.
 *                        'path' can be located on hdfs or posix, distinguish by its prefix.
 *                        eg: 'hdfs://' means hdfs, '/' means posix, 'wfs://' means wfs
 * \param format          file format
 * \param alpha           vertex's weighted for partition, -1 means use default
 * \param use_in_degree   use in-degree instead of out degree for partition
 *
 * \return
 *      graph structure in bp_bcsr form
 **/
template <template<typename, typename> class ECACHE = plato::edge_block_cache_t, template<typename> class VCACHE = plato::vertex_cache_t>
std::shared_ptr<bp_bcsr_t> create_bp_bcsr_seq_from_path(
    plato::graph_info_t*  pgraph_info,
    const std::string&    path,
    plato::edge_format_t  format,
    int                   alpha = -1,
    bool                  use_in_degree = false) {

  using partition_t = bp_bcsr_t::partition_t;

  auto& cluster_info = plato::cluster_info_t::get_instance();
  plato::stop_watch_t watch;

  watch.mark("t0");
  watch.mark("t1");

  auto pvcache = load_factors_cache<VCACHE>(path, format);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "load factors cache cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  auto pecache = load_edges_cache_from_factors_cache<ECACHE>(pgraph_info, *pvcache);
  
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
      degrees = plato::generate_dense_in_degrees<plato::vid_t>(*pgraph_info, *pecache);
    } else {
      degrees = plato::generate_dense_out_degrees<plato::vid_t>(*pgraph_info, *pecache);
    }

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "generate degrees cost: " << watch.show("t1") / 1000.0 << "s";
    }
    watch.mark("t1");

    plato::eid_t __edges = pgraph_info->edges_;
    // factor graph is undirected
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

  plato::graph_info_t graph_info;
  auto bp_bcsr = create_bp_bcsr_seq_from_path(
    &graph_info, FLAGS_input_factors, plato::edge_format_t::CSV,
    FLAGS_alpha, FLAGS_part_by_in
  );

  plato::algo::bp_opts_t opts;
  opts.iteration_ = FLAGS_iterations;
  opts.eps_       = FLAGS_eps;

  auto beliefs = plato::algo::belief_propagation(*bp_bcsr, graph_info, opts);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "belief propagation calculation done: " << watch.show("t0") / 1000.0 << "s";
  }

  watch.mark("t0");
  {
    plato::thread_local_fs_output os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), true);

    beliefs.foreach<int> (
      [&](plato::vid_t v_i, bp_dist_t* dval) {
        auto& fs_output = os.local();
        if (dval->values_) {
          fs_output << v_i << "," << *dval << "\n";
        }
        return 0;
      }
    );
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "save result cost: " << watch.show("t1") / 1000.0 << "s";
  } 

  return 0;
}