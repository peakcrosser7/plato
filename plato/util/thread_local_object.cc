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

#include <future>

#include "thread_local_object.h"
#include "defer.hpp"

namespace plato {

namespace thread_local_object_detail {

struct dist_obj;
struct local_obj;

/// @brief 全局对象容量
static constexpr int objs_capacity = 4096;
/// @brief 全局对象锁
static std::mutex g_mutex_;

/// @brief 分布式对象数组(进程内均可见)
static std::vector<std::shared_ptr<dist_obj>> dist_objs_(objs_capacity);
/// @brief 每个对象的互斥锁数组
static std::vector<std::mutex> mutex_v_(objs_capacity);
/// @brief 线程本地对象数组
static thread_local std::vector<local_obj> local_objs_(objs_capacity);

/// @brief 线程本地对象
struct local_obj {
  /// @brief 线程本地的内部封装对象
  std::shared_ptr<void> local_obj_p_;
  /// @brief 所在的分布式对象
  std::shared_ptr<dist_obj> dist_obj_;
  /// @brief boost hook,用于链接链表
  boost::intrusive::list_member_hook<> link_;
  ~local_obj();
};

/// @brief 分布式对象
struct dist_obj : public std::enable_shared_from_this<dist_obj> {
  // `boost::intrusive::list`:侵入式列表,有元素自身维护链接关系
  using objs_list_t = boost::intrusive::list<local_obj, boost::intrusive::member_hook<local_obj, boost::intrusive::list_member_hook<>, &local_obj::link_>>;

  /// @brief 用于创建内部封装对象的构造函数
  std::function<void* ()> construction_;
  /// @brief 用于创建内部封装对象的析构函数
  std::function<void(void *)> destruction_;
  /// @brief 正在关闭标识符
  bool closing_ = false;
  /// @brief 关联的线程本地对象的侵入式列表
  objs_list_t objs_list_;
  /// @brief 关联的内部封装对象的列表
  std::list<std::shared_ptr<void>> objs_p_list_;

  /// @brief promise对象
  std::promise<void> pro_;
  /// @brief 与promise对象绑定的future异步对象
  std::future<void> fut_ = pro_.get_future();

  dist_obj(std::function<void* ()> construction, std::function<void(void *)> destruction) :
    construction_(std::move(construction)), destruction_(std::move(destruction)) { }

  ~dist_obj() {
    // 清空列表
    objs_p_list_.clear();
    // 设置异步操作值
    pro_.set_value();
  }
};

local_obj::~local_obj() {
  int id = this - local_objs_.data(); // 当前对象ID
  std::lock_guard<std::mutex> lock(mutex_v_[id]);
  if (link_.is_linked()) {  // 该对象链接在列表中
    CHECK(dist_obj_ && local_obj_p_);
    local_obj_p_.reset();
    // 从列表中删除
    dist_obj_->objs_list_.erase(dist_obj_->objs_list_.iterator_to(*this));
    dist_obj_.reset();
  }
}

/// @brief 创建分布式当前节点的全局作用域对象
/// @param construction 对象构造函数
/// @param destruction 对象析构函数
/// @return 对象ID
int create_object(std::function<void*()> construction, std::function<void(void *)> destruction) {
  std::lock_guard<std::mutex> lock(g_mutex_);
  for (int id = 0; id < objs_capacity; ++id) {
    if (!dist_objs_[id]) {  // 对象位置为空则在此处创建对象
      dist_objs_[id] = std::make_shared<dist_obj>(std::move(construction), std::move(destruction));
      return id;
    }
  }
  return -1;
}

/// @brief 删除分布式对象及其关联的线程本地对象
/// @param id 对象ID
void delete_object(int id) {
  if (id < 0 || id >= objs_capacity) throw std::runtime_error("invalid object id");
  std::shared_ptr<dist_obj> dist_obj_ = dist_objs_[id];
  if (!dist_obj_) throw std::runtime_error("object not exist.");

  {
    std::lock_guard<std::mutex> lock(mutex_v_[id]);
    // 标记正在关闭对象
    dist_obj_->closing_ = true;

    // 将线程本地对象从列表中移除
    while (!dist_obj_->objs_list_.empty()) {
      local_obj& obj = dist_obj_->objs_list_.front();
      obj.local_obj_p_.reset();
      obj.dist_obj_.reset();
      dist_obj_->objs_list_.pop_front();
    }
    // 重置对象
    dist_objs_[id].reset();
  }

  std::future<void> fut = std::move(dist_obj_->fut_);
  // 重置分布式对象
  dist_obj_.reset();
  fut.get();
}

/// @brief 返回分布式对象关联的线程本地的对象
/// @param id 分布式对象ID
/// @return 线程本地的内部封装对象
void* get_local_object(int id) {
  // 线程本地对象数组
  static thread_local local_obj* local_objs_p_;
  //  `__glibc_unlikely()`:提示编译器某个条件分支不太可能发生
  if (__glibc_unlikely(!local_objs_p_)) {
    // local_objs_ has a non-trivial constructor which means that the compiler
    // must make sure the instance local to the current thread has been
    // constructed before each access.
    // Unfortunately, this means that GCC will emit an unconditional call
    // to __tls_init(), which may incurr a noticeable overhead.
    // This can be solved by adding an easily predictable branch checking
    // whether the object has already been constructed.
    local_objs_p_ = local_objs_.data();
  }

  local_obj& obj = local_objs_p_[id];   // 分布式对象ID对应的线程本地对象
  if (__glibc_unlikely(!obj.local_obj_p_.get())) {  // 本地对象为空时
    std::shared_ptr<dist_obj> dist_obj_ = dist_objs_[id];
    if (!dist_obj_) throw std::runtime_error("object not exist.");
    // 构建内部封装的对象(由于get_local_object()应该是每个线程调用,因此此处每个线程都会构建一个对象)
    std::shared_ptr<void> p(dist_obj_->construction_(), dist_obj_->destruction_);
    {
      std::lock_guard<std::mutex> lock(mutex_v_[id]);
      if (dist_obj_->closing_) throw std::runtime_error("dist_obj is closing.");
      CHECK(!obj.link_.is_linked() && !obj.local_obj_p_ && !obj.dist_obj_);
      // 添加线程本地对象到侵入式列表
      dist_obj_->objs_list_.push_back(obj);
      // 添加内部封装对象到列表
      dist_obj_->objs_p_list_.push_back(p);
      // 设置线程本地对象的内部封装对象和所在的分布式对象
      obj.local_obj_p_ = p;
      obj.dist_obj_ = dist_obj_;
    }
  }

  return obj.local_obj_p_.get();
}

unsigned objects_num() {
  std::lock_guard<std::mutex> lock(g_mutex_);
  size_t num = 0;
  for (std::shared_ptr<dist_obj> dist_obs_ : dist_objs_) {
    if (dist_obs_) {
      num++;
    }
  }
  return num;
}

unsigned objects_num(int id) {
  if (id < 0 || id >= objs_capacity) throw std::runtime_error("invalid object id");
  std::shared_ptr<dist_obj> dist_obs_ = dist_objs_[id];
  std::lock_guard<std::mutex> lock(mutex_v_[id]);
  return dist_obs_->objs_p_list_.size();
}

void object_foreach(int id, std::function<void(void*)> reducer) {
  if (id < 0 || id >= objs_capacity) throw std::runtime_error("invalid object id");
  std::shared_ptr<dist_obj> dist_obj_ = dist_objs_[id];
  if (!dist_obj_) throw std::runtime_error("object not exist.");

  std::lock_guard<std::mutex> lock(mutex_v_[id]);
  for (std::shared_ptr<void>& p : dist_obj_->objs_p_list_) {
    reducer(p.get());
  }
}

}

}
