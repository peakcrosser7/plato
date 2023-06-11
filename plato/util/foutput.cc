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

#include "foutput.h"

#include "boost/algorithm/string.hpp"
#include "boost/iostreams/device/file.hpp"
#include "boost/filesystem.hpp"

namespace plato {

/// @brief 创建文件夹
/// @param path 文件夹路径
static void touch_dir(const std::string& path) {
  if (!boost::filesystem::exists(path)) { // 文件夹不存在则创建
      boost::filesystem::create_directories(path);
  }
  CHECK(boost::filesystem::is_directory(path));
}

fs_mt_omp_output_t::fs_mt_omp_output_t(const std::string& path, const std::string& prefix, bool compressed) {
  int threads = 0;

  #pragma omp parallel
  {
    #pragma omp single
    { threads = omp_get_num_threads(); }
  }

  fs_v_.resize(threads);
  fs_output_v_.resize(threads);

  if (boost::istarts_with(path, "hdfs://")) {
    for (int i = 0; i < threads; ++i) {
      if (compressed) {
        std::string filename = (boost::format("%s/%s%04d.csv.gz") % path.c_str() % prefix.c_str() % i).str();
        fs_v_[i].reset(new plato::hdfs_t::fstream(plato::hdfs_t::get_hdfs(filename), filename, true));
        fs_output_v_[i].reset(new boost::iostreams::filtering_stream<boost::iostreams::output>());
        fs_output_v_[i]->push(boost::iostreams::gzip_compressor());
        fs_output_v_[i]->push(*fs_v_[i]);
      } else {
        std::string filename = (boost::format("%s/%s%04d.csv") % path.c_str() % prefix.c_str() % i).str();
        fs_v_[i].reset(new plato::hdfs_t::fstream(plato::hdfs_t::get_hdfs(filename), filename, true));
        fs_output_v_[i].reset(new boost::iostreams::filtering_stream<boost::iostreams::output>());
        fs_output_v_[i]->push(*fs_v_[i]);
      }
    }
  } else {
    touch_dir(path);
    for (int i = 0; i < threads; ++i) {
      fs_output_v_[i].reset(new boost::iostreams::filtering_stream<boost::iostreams::output>());
      if (compressed) {
        std::string filename = (boost::format("%s/%s%04d.csv.gz") % path.c_str() % prefix.c_str() % i).str();
        fs_output_v_[i]->push(boost::iostreams::gzip_compressor());
        fs_output_v_[i]->push(boost::iostreams::file_sink(filename));
      } else {
        std::string filename = (boost::format("%s/%s%04d.csv") % path.c_str() % prefix.c_str() % i).str();
        fs_output_v_[i]->push(boost::iostreams::file_sink(filename));
      }
    }
  }
}

boost::iostreams::filtering_stream<boost::iostreams::output>& fs_mt_omp_output_t::ostream(int thread_id) {
  CHECK(thread_id < (int)fs_output_v_.size());
  return *fs_output_v_[thread_id];
}

boost::iostreams::filtering_stream<boost::iostreams::output>& fs_mt_omp_output_t::ostream(void) {
  return ostream(omp_get_thread_num());
}

/// @brief 构造函数
/// @param path 文件目录
/// @param prefix 文件名前缀
/// @param compressed 是否压缩文件
thread_local_fs_output::thread_local_fs_output(const std::string& path, const std::string& prefix, bool compressed) {
  std::shared_ptr<std::atomic<unsigned>> suffix(new std::atomic<unsigned>());

  // 文件流构建函数,返回构建的文件流
  std::function<void*()> construction([path, prefix, compressed, suffix] {
    auto fs_ = new fs();

    std::string filename;
    if (compressed) { // 压缩文件
      filename = (boost::format("%s/%s%04d.csv.gz") % path % prefix % (*suffix)++).str();
      fs_->os_.push(boost::iostreams::gzip_compressor()); // 添加gzip压缩器
    } else {
      filename = (boost::format("%s/%s%04d.csv") % path % prefix % (*suffix)++).str();
    }
    fs_->filename_ = filename;

    if (boost::istarts_with(filename, "hdfs://")) {   // HDFS文件
      if (hdfs_t::get_hdfs(filename).exists(filename)) {  // 文件存在则删除
        CHECK(0 == hdfs_t::get_hdfs(filename).remove(filename, 0)) << "failed to remove exist file: " << filename;
      }
      fs_->hdfs_.reset(new hdfs_t::fstream(plato::hdfs_t::get_hdfs(filename), filename, true));
      fs_->os_.push(*fs_->hdfs_); // 添加HDFS处理
    } else {  // 普通文件
      touch_dir(path);  // 创建文件夹
      namespace fs = boost::filesystem;
      if (fs::exists(filename)) { // 文件存在则删除
        CHECK(fs::remove(filename)) << "failed to remove exist file: " << filename;
      }
      fs_->os_.push(boost::iostreams::file_sink(filename));
    }

    return (void*)fs_;
  });

  // 文件流析构函数
  std::function<void(void *)> destruction([] (void* p) {
    auto fs_ = (fs*)p;
    fs_->os_.flush(); // 文件流清空缓冲区写入文件
    CHECK(fs_->os_.good()) << "flush file filed, filename: " << fs_->filename_;
    delete fs_;
  });

  // 创建对象得到对应ID
  id_ = thread_local_object_detail::create_object(std::move(construction), std::move(destruction));
  if (-1 == id_) throw std::runtime_error("thread_local_object_detail::create_object failed.");
}

thread_local_fs_output::~thread_local_fs_output() {
  if (id_ != -1) {
    thread_local_object_detail::delete_object(id_);
    id_ = -1;
  }
}

void thread_local_fs_output::foreach(std::function<void(const std::string& filename, boost::iostreams::filtering_ostream& os)> reducer) {
  thread_local_object_detail::object_foreach(id_, [&reducer] (void* p) {
    auto& fs_ = *(fs*)p;
    reducer(fs_.filename_, fs_.os_);
  });
}

}  // namespace plato

