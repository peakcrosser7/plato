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

#pragma once

#include <list>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "glog/logging.h"

namespace plato {

/// @brief 后台执行器
class background_executor : public std::enable_shared_from_this<background_executor> {
  /// @brief 任务函数队列
  std::list<std::function<void()>> items_;
  /// @brief 任务函数队列的互斥锁
  std::mutex mutex_;
  /// @brief 线程列表的条件变量
  std::condition_variable cv_;
  /// @brief 线程刷新标志位
  bool flushing_ = false;
  /// @brief 线程列表
  std::list<std::thread> threads_;
  /// @brief 最大线程数
  size_t max_threads_;
public:
  background_executor(const background_executor&) = delete;
  background_executor& operator=(const background_executor&) = delete;
  background_executor(background_executor&&) = delete;
  background_executor& operator=(background_executor&&) = delete;

  explicit background_executor(size_t max_threads = omp_get_num_threads()) :
    max_threads_(std::max(max_threads, size_t(1))) {};

  ~background_executor() { flush(); }

  /// @brief 提交任务
  /// @param func 任务函数
  void submit(std::function<void()> func) {
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK(!flushing_);

    // 线程列表为空 或者 任务列表不为空而线程列表未达到最大线程数
    if (threads_.empty() || (!items_.empty() && threads_.size() < max_threads_)) {
      // 添加一个线程到线程队列执行
      threads_.emplace_back([this] {
        while(true) {
          // 尝试从任务队列获取函数
          while(true) {
            std::function<void()> func;
            {
              std::unique_lock<std::mutex> lock(mutex_);
              if (items_.empty()) {
                break;
              }
              func = std::move(items_.front());
              items_.pop_front();
            }
            // 执行函数
            func();
          }

          {
            std::unique_lock<std::mutex> lock(mutex_);
            // 若在刷新线程列表但无任务则退出
            if (flushing_ && items_.empty()) break;
            // 若未在刷新线程列表且无任务则等待
            if (!flushing_ && items_.empty()) cv_.wait(lock);
          }
        }
      });
    }

    // 添加任务函数
    items_.emplace_back(std::move(func));
    // 唤醒一个线程执行
    cv_.notify_one();
  }

  /// @brief 刷新线程队列执行任务
  void flush() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // 确保当前未在刷新
      CHECK(!flushing_);
      // 设置标志位
      flushing_ = true;
      // 通知所有等待的线程执行任务
      cv_.notify_all();
    }

    // 等待所有任务线程完成
    for (auto& thread : threads_) {
      thread.join();
    }
    threads_.clear();

    {
      std::unique_lock<std::mutex> lock(mutex_);
      CHECK(flushing_);
      flushing_ = false;
    }
  }
};

}