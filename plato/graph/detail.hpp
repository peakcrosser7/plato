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

#ifndef __PLATO_GRAPH_DETAIL_HPP__
#define __PLATO_GRAPH_DETAIL_HPP__

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "boost/iostreams/device/file.hpp"

#include "omp.h"
#include "mpi.h"

#include <memory>
#include <functional>

#include "plato/graph/base.hpp"
#include "plato/parallel/shuffle.hpp"
#include "plato/util/hdfs.hpp"

namespace plato {

/*
 * 将一组文件按照大小均匀分配成指定部分
 * assign files evenly by size
 *
 * \param[out] poutput 每部分的文件名的列表(大小为parts)
 * \param files 文件列表
 * \param parts 分配的部分数量
 *
 * \return 0 -- success, else failed
 **/
inline int assign_files_even_by_size(
    std::vector<std::vector<std::string>>* poutput,
    const std::vector<std::pair<std::string, size_t>>& files, int parts) {
    // 所有文件的总大小
    size_t total_size = 0;
    for (const auto& finfo : files) {
        total_size += finfo.second;
    }

    size_t cul_size = 0;
    // 每部分大小
    size_t size_per_part = total_size / parts;

    int part = 0;
    poutput->clear();
    poutput->resize(parts);
    for (const auto& finfo : files) {
        poutput->at(part).emplace_back(finfo.first);
        cul_size += finfo.second;
        if ((cul_size > size_per_part) && (part != (parts - 1))) {
            ++part;
            cul_size = 0;
        }
    }
    return 0;
}

/// @brief 基于Linux文件系统获取本节点需要处理的文件分块列表
/// @param path 文件路径
/// @return 本节点需要处理的文件分块列表
inline std::vector<std::string> get_files_from_posix(const std::string& path) {
    namespace fs = boost::filesystem;

    int rc = 0;
    auto& cluster_info = cluster_info_t::get_instance();
    std::vector<std::string> chunks;
    std::vector<std::vector<std::string>> fchunks;

    if (0 == cluster_info.partition_id_) {  // split files evenly
        std::vector<std::pair<std::string, size_t>> files;  // 文件列表

        if (fs::is_directory(path)) {  // check path is a file or a directory
            for (fs::directory_iterator it(path);
                 it != fs::directory_iterator(); ++it) {
                files.emplace_back(std::make_pair(
                    it->path().string(), fs::file_size(it->path().string())));
            }
        } else {
            files.emplace_back(std::make_pair(path, fs::file_size(path)));
        }
        // 文件按节点分块
        if ((rc = assign_files_even_by_size(&fchunks, files,
                                            cluster_info.partitions_)) < 0) {
            LOG(ERROR) << "assign_files_even_by_size failed with code: " << rc;
            throw std::runtime_error(
                (boost::format(
                     "assign_files_even_by_size failed with code: %d") %
                 rc)
                    .str());
        }
        for (size_t i = 0; i < fchunks.size(); ++i) {
            LOG(INFO) << "partition-" << i
                      << ", file-count: " << fchunks[i].size();
        }
    }

    std::atomic<uint32_t> s_index(0);
    auto shuffle_send =
        [&](shuffle_send_callback_t<std::vector<std::string>> send) {
            if (0 == cluster_info.partition_id_) {
                for (uint32_t s_i = s_index.fetch_add(1); s_i < fchunks.size();
                     s_i = s_index.fetch_add(1)) {
                    send(s_i, fchunks[s_i]);
                }
            }
        };

    auto shuffle_recv =
        [&](int /*p_i*/,
            plato::shuffle_recv_pmsg_t<std::vector<std::string>>& pmsg) {
            chunks = *pmsg;
        };

    if (0 !=
        (rc = shuffle<std::vector<std::string>>(shuffle_send, shuffle_recv))) {
        LOG(ERROR) << "shuffle failed with code: " << rc;
        throw std::runtime_error(
            (boost::format("shuffle failed with code: %d") % rc).str());
    }

    return chunks;
}

/// @brief 基于HDFS获取本节点需要处理的文件分块列表
/// @param path HDFS文件路径
/// @return 本节点需要处理的文件分块列表
inline std::vector<std::string> get_files_from_hdfs(const std::string& path) {
    int rc = 0;
    auto& cluster_info = cluster_info_t::get_instance();
    // 接收到的数据本节点的文件分块列表
    std::vector<std::string> chunks;
    // 每个集群节点对于的文件分块列表
    std::vector<std::vector<std::string>> fchunks;

    if (0 == cluster_info.partition_id_) {
        std::vector<std::pair<std::string, size_t>> files;  // 路径下文件及大小

        int num_files = 0;
        // 获取路径下所有文件
        hdfsFileInfo* hdfs_file_list_ptr = hdfsListDirectory(
            hdfs_t::get_hdfs(path).filesystem_, path.c_str(), &num_files);
        // 记录每个文件的名称和大小
        for (int i = 0; i < num_files; ++i) {
            if (hdfs_file_list_ptr[i].mSize <= 0) {
                continue;
            }
            files.emplace_back(
                std::make_pair(std::string(hdfs_file_list_ptr[i].mName),
                               hdfs_file_list_ptr[i].mSize));
        }
        hdfsFreeFileInfo(hdfs_file_list_ptr, num_files);
        // 将文件按照集群结点数均匀划分
        if ((rc = assign_files_even_by_size(&fchunks, files,
                                            cluster_info.partitions_)) < 0) {
            LOG(ERROR) << "assign_files_even_by_size failed with code: " << rc;
            throw std::runtime_error(
                (boost::format("assign_files_even_by_size failed with code: %d") % rc).str());
        }
        for (size_t i = 0; i < fchunks.size(); ++i) {
            LOG(INFO) << "partition-" << i
                      << ", file-count: " << fchunks[i].size();
        }
    }

    // assign file chunks
    std::atomic<uint32_t> s_index(0);
    // 洗牌的发送任务
    auto shuffle_send =
        [&](shuffle_send_callback_t<std::vector<std::string>> send) {
            // 0 号进程发送每个节点对应的文件列表
            if (0 == cluster_info.partition_id_) {
                for (uint32_t s_i = s_index.fetch_add(1); s_i < fchunks.size();
                     s_i = s_index.fetch_add(1)) {
                    send(s_i, fchunks[s_i]);
                }
            }
        };

    auto shuffle_recv = [&](int /*p_i*/,
            plato::shuffle_recv_pmsg_t<std::vector<std::string>>& pmsg) {
            chunks = *pmsg;
        };

    // 洗牌,即发送至每个节点对应的文件列表并接收
    if (0 !=
        (rc = shuffle<std::vector<std::string>>(shuffle_send, shuffle_recv))) {
        LOG(ERROR) << "shuffle failed with code: " << rc;
        throw std::runtime_error(
            (boost::format("shuffle failed with code: %d") % rc).str());
    }

    return chunks;
}

/// @brief 获取本节点需要处理的文件分块列表
/// @param path 文件路径
inline std::vector<std::string> get_files(const std::string& path) {
    CHECK(!path.empty()) << "invalid path: " << path;

    if (boost::starts_with(path, "hdfs://")) {  // Apache Hadoop分布式文件
        return get_files_from_hdfs(path);
    } else if (boost::starts_with(path, "wfs://")) {
        CHECK(false) << "currently we does not support wfs";
        abort();
    } else {
        return get_files_from_posix(path);
    }
}

/*
 * 处理文件
 * \param filename 文件名
 * \param func   auto func(boost::iostreams::filtering_istream& is) 处理文件的回调函数
 *
 * \return 0 -- success, else failed
 **/
template <typename Func>
inline void with_file(const std::string& filename, Func func) {
    boost::iostreams::filtering_istream fin;
    if (boost::iends_with(filename, ".gz")) {
        fin.push(boost::iostreams::gzip_decompressor());
    }

    if (boost::istarts_with(filename, "hdfs://")) {
        hdfs_t::fstream hdfs_fin(hdfs_t::get_hdfs(filename), filename);
        fin.push(hdfs_fin);
        func(fin);
    } else {
        fin.push(boost::iostreams::file_source(filename));
        func(fin);
    }
}

}  // namespace plato

#endif
