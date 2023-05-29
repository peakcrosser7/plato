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

#ifndef __PLATO_ARCHIVE_HPP__
#define __PLATO_ARCHIVE_HPP__

#include <cstdint>
#include <memory>
#include <utility>

#include "yas/binary_oarchive.hpp"
#include "yas/binary_iarchive.hpp"

#include "plato/graph/base.hpp"
#include "plato/util/stream.hpp"

namespace plato {

/***************************************************************************/
// serialization helper for non-trivial class

/// @brief 用于非平凡类的输出流序列化类
/// @tparam MSG_T 序列化的消息类型
/// @tparam OSTREAM_T 输出流类型
/// @tparam ENABLE 
template <typename MSG_T, typename OSTREAM_T, typename ENABLE = void>
class oarchive_t {
  public:
    // 二进制序列化(二进制格式,主机字节序,无头部信息)
    using oarchive_detail_t = yas::binary_oarchive<OSTREAM_T, yas::binary | yas::ehost | yas::no_header>;

    /// @brief
    /// @param reserved 输出流分配的容量
    oarchive_t(size_t reserved = HUGESIZE)
        : count_(0), postream_(new OSTREAM_T(reserved)),
          poarchive_(new oarchive_detail_t(*postream_)) {}

    oarchive_t(OSTREAM_T &&os)
        : count_(0), postream_(new OSTREAM_T(std::forward<OSTREAM_T>(os))),
          poarchive_(new oarchive_detail_t(*postream_)) {}

    oarchive_t(const std::shared_ptr<OSTREAM_T> &pos)
        : count_(0), postream_(pos),
          poarchive_(new oarchive_detail_t(*postream_)) {}

    /// @brief  发出(序列化)消息到输出流
    void emit(const MSG_T &msg) {
        (*poarchive_) & msg;
        ++count_;
    }

    /// @brief 已序列化的消息数
    size_t count(void) const { return count_; }
    /// @brief 输出流大小
    size_t size(void) const { return postream_->size(); }

    void reset(void) {
        count_ = 0;
        postream_->reset();
    }

    /// @brief 获取输出流原生缓存
    intrusive_buffer_t get_intrusive_buffer(void) const {
        return postream_->get_intrusive_buffer();
    }

    /// @brief 获取输出流缓存
    std::shared_ptr<OSTREAM_T> get_stream(void) { return postream_; }

    /// @brief 是否是序列化平凡类的
    /// @return false
    bool is_trivial(void) { return false; }

  protected:
    /// @brief 已序列化的消息数
    size_t count_;
    /// @brief 输出流
    std::shared_ptr<OSTREAM_T> postream_;
    /// @brief 输出存档(yas序列化对象)
    std::shared_ptr<oarchive_detail_t> poarchive_;
};

/// @brief 用于非平凡类的输入流反序列化类
/// @tparam MSG_T 
/// @tparam ISTREAM_T 
/// @tparam ENABLE 
template <typename MSG_T, typename ISTREAM_T, typename ENABLE = void>
class iarchive_t {
  public:
    using pmsg_t = std::unique_ptr<MSG_T>;
    // 二进制反序列化(二进制格式,主机字节序,无头部信息)
    using iarchive_detail_t = yas::binary_iarchive<ISTREAM_T, yas::binary | yas::ehost | yas::no_header>;

    /// @brief 
    /// @param data 输入流缓存地址
    /// @param size 输入流缓存大小
    /// @param count 需要反序列化的消息数
    iarchive_t(const char *data, size_t size, size_t count)
        : count_(count), pistream_(new ISTREAM_T(data, size)),
          piarchive_(new iarchive_detail_t(*pistream_)) {}

    iarchive_t(ISTREAM_T &&is, size_t count)
        : count_(count), pistream_(new ISTREAM_T(std::forward<ISTREAM_T>(is))),
          piarchive_(new iarchive_detail_t(*pistream_)) {}

    /// @brief 接收(反序列化)消息
    /// @return 消息的unique_ptr
    pmsg_t absorb(void) {
        if (count_ <= 0) {
            return pmsg_t(nullptr);
        }

        pmsg_t msg(new MSG_T()); // use std::default_delete
        // 反序列化消息
        (*piarchive_) & (*msg);
        // 需要反序列化的消息数-1
        --count_;
        return msg;
    }

    /// @brief 返回需要反序列化的消息数
    size_t count(void) const { return count_; }

    /// @brief 是否是序列化平凡类的
    /// @return false
    bool is_trivial(void) { return false; }

  protected:
    /// @brief 需要反序列化的消息数
    size_t count_;
    /// @brief 输入流
    std::unique_ptr<ISTREAM_T> pistream_;
    /// @brief 输入存档(yas反序列化对象)
    std::unique_ptr<iarchive_detail_t> piarchive_;
};

/***************************************************************************/
// serialization helper for trivial class

/// @brief 用于平凡类的输出流序列化类
/// @tparam MSG_T 
/// @tparam OSTREAM_T 
template <typename MSG_T, typename OSTREAM_T>
class oarchive_t<
    MSG_T, OSTREAM_T,
    typename std::enable_if<std::is_trivial<MSG_T>::value &&
                            std::is_standard_layout<MSG_T>::value>::type> {
  public:
    oarchive_t(size_t reserved = HUGESIZE)
        : count_(0), postream_(new OSTREAM_T(reserved)) {}

    oarchive_t(OSTREAM_T &&os)
        : count_(0), postream_(new OSTREAM_T(std::forward<OSTREAM_T>(os))) {}

    oarchive_t(const std::shared_ptr<OSTREAM_T> &pos)
        : count_(0), postream_(pos) {}

    /// @brief 发出(序列化)消息到输出流
    void emit(const MSG_T &msg) {
        postream_->write(&msg, sizeof(msg));
        ++count_;
    }

    /// @brief 已序列化的消息数
    size_t count(void) const { return count_; }
    /// @brief 输出流大小
    size_t size(void) const { return postream_->size(); }

    void reset(void) {
        count_ = 0;
        postream_->reset();
    }

    /// @brief 获取输出流原生缓存
    intrusive_buffer_t get_intrusive_buffer(void) const {
        return postream_->get_intrusive_buffer();
    }

    /// @brief 获取输出流缓存
    std::shared_ptr<OSTREAM_T> get_stream(void) { return postream_; }
    
    /// @brief 是否是序列化平凡类的
    /// @return true
    bool is_trivial(void) { return true; }

  protected:
    /// @brief 已序列化的消息数
    size_t count_;
    /// @brief 输出流
    std::shared_ptr<OSTREAM_T> postream_;
};

/// @brief 用于非平凡类的输入流反序列化类
template <typename MSG_T, typename ISTREAM_T>
class iarchive_t<
    MSG_T, ISTREAM_T,
    typename std::enable_if<std::is_trivial<MSG_T>::value &&
                            std::is_standard_layout<MSG_T>::value>::type> {
  protected:
    static void del(MSG_T *) { /* empty deleter */
    }

  public:
    using pmsg_t = std::unique_ptr<MSG_T, decltype(&del)>;

    iarchive_t(const char *data, size_t size, size_t count)
        : count_(count), idx_(0), pistream_(new ISTREAM_T(data, size)),
          buffer_(pistream_->get_intrusive_buffer()) {}

    iarchive_t(ISTREAM_T &&is, size_t count)
        : count_(count), idx_(0),
          pistream_(new ISTREAM_T(std::forward<ISTREAM_T>(is))),
          buffer_(pistream_->get_intrusive_buffer()) {}

    /// @brief 接收(反序列化)消息
    /// @return 消息的unique_ptr
    pmsg_t absorb(void) {
        if (count_ <= 0) {
            return pmsg_t(nullptr, &del);
        }

        // pmsg_t msg(&((MSG_T*)buffer_.data_)[idx_], &del);
        pmsg_t msg(&((MSG_T *)buffer_.data_)[idx_], &del);
        --count_;
        ++idx_;
        return msg;
    }

    /// @brief 返回需要反序列化的消息数
    size_t count(void) const { return count_; }

    /// @brief 是否是序列化平凡类的
    /// @return true
    bool is_trivial(void) { return true; }

  protected:
    /// @brief 需要反序列化的消息数
    size_t count_;
    /// @brief 输入流待反序列化的消息的索引
    size_t idx_;
    /// @brief 输入流
    std::unique_ptr<ISTREAM_T> pistream_;
    /// @brief 输入流缓存
    intrusive_buffer_t buffer_;
};

/***************************************************************************/

} // namespace plato

#endif
