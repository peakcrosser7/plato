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

#ifndef __PLATO_UTIL_FOUTPUT_HPP__
#define __PLATO_UTIL_FOUTPUT_HPP__

#include <cstdint>
#include <cstdlib>
#include <string>
#include <memory>

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "hdfs.hpp"
#include "thread_local_object.h"

namespace plato {

class fs_mt_omp_output_t {
public:
  fs_mt_omp_output_t(const std::string& path, const std::string& prefix, bool compressed = true);

  fs_mt_omp_output_t(const fs_mt_omp_output_t&) = delete;
  fs_mt_omp_output_t& operator=(const fs_mt_omp_output_t&) = delete;

  // get ostream by current thread-id
  boost::iostreams::filtering_stream<boost::iostreams::output>& ostream(void);
  boost::iostreams::filtering_stream<boost::iostreams::output>& ostream(int thread_id);

protected:
  std::vector<std::unique_ptr<hdfs_t::fstream>> fs_v_;
  std::vector<std::unique_ptr<boost::iostreams::filtering_stream<boost::iostreams::output>>>
    fs_output_v_;
};

/// @brief 线程本地文件输出流
class thread_local_fs_output {
public:
  thread_local_fs_output(const thread_local_fs_output&) = delete;
  thread_local_fs_output& operator=(const thread_local_fs_output&) = delete;
  thread_local_fs_output(thread_local_fs_output&& x) noexcept : id_(x.id_) {
    x.id_ = -1;
  }
  thread_local_fs_output& operator=(thread_local_fs_output &&x) noexcept {
    if (this != &x) {
      this->~thread_local_fs_output();
      new(this) thread_local_fs_output(std::move(x));
    }
    return *this;
  }

  thread_local_fs_output(const std::string& path, const std::string& prefix, bool compressed = true);

  ~thread_local_fs_output();

  void foreach(std::function<void(const std::string& filename, boost::iostreams::filtering_ostream& os)> reducer);

  /// @brief 获取线程本地的文件输出流
  [[gnu::always_inline]] [[gnu::hot]] // `[[gnu::hot]]`:标记函数为热点函数,即被频繁调用的函数
  boost::iostreams::filtering_ostream& local() { return ((fs*)thread_local_object_detail::get_local_object(id_))->os_; }
protected:
  /// @brief 文件流
  struct fs {
    /// @brief 文件名
    std::string filename_;
    /// @brief HDFS文件流
    std::unique_ptr<hdfs_t::fstream> hdfs_;
    /// @brief 输出流
    boost::iostreams::filtering_ostream os_;
  };

  /// @brief 全局文件流ID
  int id_;
};

/*
 * \param filename  file name
 * \param func      auto func(boost::iostreams::filtering_ostream& os)
 *
 * \return 0 -- success, else failed
 **/
template <typename Func>
inline void with_ofile(const std::string& filename, Func&& func) {
  boost::iostreams::filtering_ostream fout;
  if (boost::iends_with(filename, ".gz")) {
    fout.push(boost::iostreams::gzip_compressor());
  }

  if (boost::istarts_with(filename, "hdfs://")) {
    hdfs_t::fstream hdfs_fout(hdfs_t::get_hdfs(filename), filename, true);
    fout.push(hdfs_fout);
    func(fout);
    fout.reset();  // trigger filtering_ostream flush before fstream been closed
  } else {
    fout.push(boost::iostreams::file_sink(filename));
    func(fout);
    fout.reset();  // trigger filtering_ostream flush before fstream been closed
  }
}

}  // namespace plato

#endif

