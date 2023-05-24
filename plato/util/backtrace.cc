/*
 * This file is open source software, licensed to you under the terms
 * of the Apache License, Version 2.0 (the "License").  See the NOTICE file
 * distributed with this work for additional information regarding copyright
 * ownership.  You may not use this file except in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * Copyright 2017 ScyllaDB
 */
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

#include "backtrace.h"

#include <link.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

#include <errno.h>
#include <string.h>

namespace plato {

static int dl_iterate_phdr_callback(struct dl_phdr_info* info,
                                    size_t /* size */, void* data) {
    std::size_t total_size{0};
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const auto hdr = info->dlpi_phdr[i];

        // Only account loadable, executable (text) segments
        if (hdr.p_type == PT_LOAD && (hdr.p_flags & PF_X) == PF_X) {
            total_size += hdr.p_memsz;
        }
    }

    reinterpret_cast<std::vector<shared_object>*>(data)->push_back(
        {info->dlpi_name, info->dlpi_addr, info->dlpi_addr + total_size});

    return 0;
}

static std::vector<shared_object> enumerate_shared_objects() {
    std::vector<shared_object> shared_objects;
    dl_iterate_phdr(dl_iterate_phdr_callback, &shared_objects);

    return shared_objects;
}

static const std::vector<shared_object> shared_objects{
    enumerate_shared_objects()};
static const shared_object uknown_shared_object{
    "", 0, std::numeric_limits<uintptr_t>::max()};

bool operator==(const frame& a, const frame& b) {
    return a.so == b.so && a.addr == b.addr;
}

frame decorate(uintptr_t addr) {
    char** s = backtrace_symbols((void**)&addr, 1);
    std::string symbol(*s);
    free(s);

    // If the shared-objects are not enumerated yet, or the enumeration
    // failed return the addr as-is with a dummy shared-object.
    if (shared_objects.empty()) {
        return {&uknown_shared_object, addr, std::move(symbol)};
    }

    auto it = std::find_if(shared_objects.begin(), shared_objects.end(),
                           [&](const shared_object& so) {
                               return addr >= so.begin && addr < so.end;
                           });

    // Unidentified addresses are assumed to originate from the executable.
    auto& so = it == shared_objects.end() ? shared_objects.front() : *it;
    return {&so, addr - so.begin, std::move(symbol)};
}

saved_backtrace current_backtrace() noexcept {
    saved_backtrace::vector_type v;
    backtrace([&](frame f) {
        if (v.size() < v.capacity()) {
            v.emplace_back(std::move(f));
        }
    });
    return saved_backtrace(std::move(v));
}

size_t saved_backtrace::hash() const {
    size_t h = 0;
    for (auto f : _frames) {
        h = ((h << 5) - h) ^ (f.so->begin + f.addr);
    }
    return h;
}

std::ostream& operator<<(std::ostream& out, const saved_backtrace& b) {
    for (auto f : b._frames) {
        out << "  ";
        if (!f.so->name.empty()) {
            out << f.so->name << "+";
        }
        out << boost::format("0x%x\n") % f.addr;
    }
    return out;
}

// Installs handler for Signal which ensures that Func is invoked only once
// in the whole program and that after it is invoked the default handler is
// restored.
/// @brief 安装一次性信号处理程序
/// @tparam Signal 信号编号
/// @tparam Func 处理函数
template <int Signal, void (*Func)()>
void install_oneshot_signal_handler() {
    // 用于确保信号处理函数Func只被调用1次
    static bool handled = false;
    static std::mutex lock;

    struct sigaction sa;
    // 定义信号处理程序
    sa.sa_sigaction = [](int sig, siginfo_t* /* info */, void* /* p */) {
        // 作用域锁
        std::lock_guard<std::mutex> g(lock);
        if (!handled) {
            handled = true;
            // 信号处理程序
            Func();
            // 设置信号为默认处理程序
            signal(sig, SIG_DFL);
        }
    };
    // 设置为所有信号掩码,以确保在号处理程序执行期间不会被其他信号中断
    sigfillset(&sa.sa_mask);
    // 使用sa_sigaction作为信号处理程序，并在系统调用被信号中断时自动重启系统调用
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    if (Signal == SIGSEGV) {
        // 在另一个堆栈上执行信号处理程序
        sa.sa_flags |= SA_ONSTACK;
    }
    // 将信号处理程序安装到指定的信号上
    auto r = ::sigaction(Signal, &sa, nullptr);
    if (r == -1) {
        throw std::system_error();
    }
}

/// @brief 输出回溯信息
/// @param buf 回溯信息缓存
static void print_with_backtrace(backtrace_buffer& buf) noexcept {
    buf.append(".\nBacktrace:\n");
    buf.append_backtrace();
    buf.flush();
}

/// @brief 输出回溯信息
/// @param cause 回溯造成的原因
static void print_with_backtrace(const char* cause) noexcept {
    backtrace_buffer buf;
    // 添加原因
    buf.append(cause);
    print_with_backtrace(buf);
}

/// @brief 信号处理函数
static void sigsegv_action() noexcept {
    print_with_backtrace("Segmentation fault");
}

static void sigabrt_action() noexcept { print_with_backtrace("Aborting"); }

/// @brief 安装一次性信号处理函数
void install_oneshot_signal_handlers() {
    // Mask most, to prevent threads (esp. dpdk helper threads)
    // from servicing a signal.  Individual reactors will unmask signals
    // as they become prepared to handle them.
    //
    // We leave some signals unmasked since we don't handle them ourself.

    // 信号集
    sigset_t sigs;
    // 初始化包含所有信号
    sigfillset(&sigs);
    // 将特定信号删除
    for (auto sig : {SIGHUP, SIGQUIT, SIGILL, SIGABRT, SIGFPE, SIGSEGV, SIGALRM,
                     SIGCONT, SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU}) {
        sigdelset(&sigs, sig);
    }
    // 设置信号掩码
    // 设置信号集中的信号被阻塞,这些信号不会被线程接收
    pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

    // 添加信号SIGSEGV,SIGABRT的信号处理函数
    install_oneshot_signal_handler<SIGSEGV, sigsegv_action>();
    install_oneshot_signal_handler<SIGABRT, sigabrt_action>();
}

}  // namespace plato
