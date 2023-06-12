/*
  Tencent is pleased to support the open source community by making
  Plato available.
  Copyright (C) 2019 THL A29 Limited, a Tencent company.
  All rights reserved.

  Licensed under the BSD 3-Clause License (the "License"); you may
  not use this file except in compliance with the License. You may
  obtain a copy of the License at

  https://opensource.org/licenses/BSD-3-Clause

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" basis,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied. See the License for the specific language governing
  permissions and limitations under the License.

  See the AUTHORS file for names of contributors.
*/

#include <cmath>
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

DEFINE_string(input,       "",      "input file, in csv format, without edge data");
DEFINE_string(output,      "",      "output directory");
DEFINE_bool(is_directed,   false,   "is graph directed or not");
DEFINE_bool(part_by_in,    false,   "partition by in-degree");
DEFINE_int32(alpha,        -1,      "alpha value used in sequence balance partition");
DEFINE_uint64(iterations,  100,     "number of iterations");
DEFINE_double(damping,     0.85,    "the damping factor");  // 阻尼系数
DEFINE_double(eps,         0.001,   "the calculation will be consider \
                                      as complete if the difference of PageRank values between iterations \
                                      change less than this value for every node");   // 收敛差值epsilon

bool string_not_empty(const char*, const std::string& value) {
  if (0 == value.length()) { return false; }
  return true;
}

DEFINE_validator(input,  &string_not_empty);
DEFINE_validator(output, &string_not_empty);

void init(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
}

int main(int argc, char** argv) {
  plato::stop_watch_t watch;
  auto& cluster_info = plato::cluster_info_t::get_instance();

  init(argc, argv);
  cluster_info.initialize(&argc, &argv);

  watch.mark("t0");

  // init graph
  plato::graph_info_t graph_info(FLAGS_is_directed);
  // 本地DCSC图结构
  auto pdcsc = plato::create_dcsc_seqs_from_path<plato::empty_t>(
    &graph_info, FLAGS_input, plato::edge_format_t::CSV,
    plato::dummy_decoder<plato::empty_t>, FLAGS_alpha, FLAGS_part_by_in
  );

  using graph_spec_t         = std::remove_reference<decltype(*pdcsc)>::type;
  using partition_t          = graph_spec_t::partition_t;
  using adj_unit_list_spec_t = graph_spec_t::adj_unit_list_spec_t;
  using rank_state_t         = plato::dense_state_t<double, partition_t>;

  // init state
  // 此轮迭代结点PageRank值稠密数据
  std::shared_ptr<rank_state_t> curt_rank(new rank_state_t(graph_info.max_v_i_, pdcsc->partitioner()));
  // 下一轮迭代结点PageRank值稠密数据
  std::shared_ptr<rank_state_t> next_rank(new rank_state_t(graph_info.max_v_i_, pdcsc->partitioner()));

  watch.mark("t1");
  // 结点出度稠密数据
  auto odegrees = plato::generate_dense_out_degrees_fg<uint32_t>(graph_info, *pdcsc, false);

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "generate out-degrees from graph cost: " << watch.show("t1") / 1000.0 << "s";
  }
  watch.mark("t1");

  watch.mark("t2"); // do computation

  /**
   * PageRank算法说明:
   * 使用公式: PR(A)=(1-d)+d(PR(T_1)/C(T_1)+...+PR(T_n)/C(T_n)) --Ref from Google
   * 其中:
   *  RP(A):结点A的PageRank值
   *  d:阻尼系数,默认0.85
   *  T_1,...,T_n:具有指向结点A的出边的结点,即结点A的入边邻结点
   *  C(A):结点A的出边数,即结点A的出度
   * 
   * 以下代码实现中会做一个预处理:
   * 每轮迭代计算PageRank值之后,会提前计算中间值PR'(A)=PR(A)/C(A),使得下一轮计算时可以直接使用,
   * 即原PageRank公式变为: PR(A)=(1-d)+d(PR'(T_1)+...+PR'(T_n))
   * 这也使得除去最后一轮迭代,cur_rank中记录的是PR'(A)=PR(A)/C(A)而非PR(A)
  */

  // 两轮所有结点的PageRank值的差值之和
  double delta = curt_rank->foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
      // 初始化每个结点PageRank值为1.
      *pval = 1.0;
      if (odegrees[v_i] > 0) {
        // 为下一轮迭代计算PR'(A)=PR(A)/C(A)
        *pval = *pval / odegrees[v_i];
      }
      return 1.0;
    }
  );

  using context_spec_t = plato::mepa_ag_context_t<double>;
  using message_spec_t = plato::mepa_ag_message_t<double>;

  // 迭代指定轮数
  for (uint32_t epoch_i = 0; epoch_i < FLAGS_iterations; ++epoch_i) {
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "delta: " << delta;
    }

    watch.mark("t1");
    next_rank->fill(0.0); // 初始化下一轮迭代PageRank值为0.

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "] init-next cost: "
        << watch.show("t1") / 1000.0 << "s";
    }

    watch.mark("t1");
    // PULL模式 聚合计算
    plato::aggregate_message<double, int, graph_spec_t> (*pdcsc,
      // 消息发送函数
      [&](const context_spec_t& context, plato::vid_t v_i, const adj_unit_list_spec_t& adjs) {
        // v_i入边邻结点此轮PageRank值(PR')之和
        double rank_sum = 0.0;
        // 遍历v_i的每个入边邻结点
        for (auto it = adjs.begin_; adjs.end_ != it; ++it) {
          // 累加邻结点的PR'值
          rank_sum += (*curt_rank)[it->neighbour_];
        }
        // 发送结点及其入边邻结点PR'值之和
        context.send(message_spec_t { v_i, rank_sum });
      },
      // 消息接收函数
      [&](int /*p_i*/, message_spec_t& msg) {
        // 累加v_i全局的入边邻结点PR'值之和
        plato::write_add(&(*next_rank)[msg.v_i_], msg.message_);
        return 0;
      }
    );

    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "] message-passing cost: "
        << watch.show("t1") / 1000.0 << "s";
    }

    watch.mark("t1");
    if (FLAGS_iterations - 1 == epoch_i) {  // 最后一轮迭代
      // 计算每个结点的PageRank值并返回差值
      delta = next_rank->foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          // PR(v_i)=1-d+d(\sum_{(v_i,v_j)\in E}PR'(v_j))=1-d+d(\sum_{(v_i,v_j)\in E}PR(v_j)/OutDegrees(v_j))
          *pval = 1.0 - FLAGS_damping + FLAGS_damping * (*pval);
          return 0;
        }
      );
    } else {  // 不为最后一轮迭代
      // 计算每个结点的PageRank值并返回差值
      delta = next_rank->foreach<double> (
        [&](plato::vid_t v_i, double* pval) {
          *pval = 1.0 - FLAGS_damping + FLAGS_damping * (*pval);
          // 为下一轮迭代预处理,即计算中间值PR'(v_i)=PR(v_i)/OutDegrees(v_i)
          // 出度为0的结点不处理是因为没有结点会根据其计算PageRank值
          if (odegrees[v_i] > 0) {
            *pval = *pval / odegrees[v_i];
            // 返回两轮PageRank值的差值(由于做了预处理因此需要再乘上出度)
            return fabs(*pval - (*curt_rank)[v_i]) * odegrees[v_i];
          }
          return fabs(*pval - (*curt_rank)[v_i]);
        }
      );

      // 若两轮PageRank的差值小于epsilon则再进行一轮迭代就终止
      if (FLAGS_eps > 0.0 && delta < FLAGS_eps) {
        epoch_i = FLAGS_iterations - 2;
      }
    }
    if (0 == cluster_info.partition_id_) {
      LOG(INFO) << "[epoch-" << epoch_i  << "] foreach_vertex cost: "
        << watch.show("t1") / 1000.0 << "s";
    }
    std::swap(curt_rank, next_rank);
  }

  // 最终结点PageRank值之和
  delta = curt_rank->foreach<double> (
    [&](plato::vid_t v_i, double* pval) {
      return *pval;
    }
  );

  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "iteration done, cost: " << watch.show("t2") / 1000.0 << "s, rank-sum: " << delta;
    LOG(INFO) << "whole cost: " << watch.show("t0") / 1000.0 << "s";
  }

  watch.mark("t1");
  {  // save result to hdfs
    plato::thread_local_fs_output os(FLAGS_output, (boost::format("%04d_") % cluster_info.partition_id_).str(), true);
    // 输出PageRank值到文件
    curt_rank->foreach<int> (
      [&](plato::vid_t v_i, double* pval) {
        // 线程本地的文件输出流(foreach是多线程调用)
        auto& fs_output = os.local();
        fs_output << v_i << "," << *pval << "\n";
        return 0;
      }
    );
  }
  if (0 == cluster_info.partition_id_) {
    LOG(INFO) << "save result cost: " << watch.show("t1") / 1000.0 << "s";
  }

  // 统计内存使用情况
  plato::mem_status_t mstatus;
  plato::self_mem_usage(&mstatus);
  LOG(INFO) << "memory usage: " << (double)mstatus.vm_rss / 1024.0 << " MBytes";

  return 0;
}

