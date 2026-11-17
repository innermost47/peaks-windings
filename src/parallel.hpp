// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#pragma once

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace pw
{

    // Runs fn(index, worker) for index in [0, n) on at most `threads` workers. Exceptions are rethrown on the caller.
    template <class F>
    void parallel_for(int n, int threads, F &&fn)
    {
        threads = std::max(1, std::min(threads, n));
        if (threads == 1)
        {
            for (int i = 0; i < n; ++i)
                fn(i, 0);
            return;
        }
        std::atomic<int> next{0};
        std::exception_ptr error;
        std::mutex error_mutex;
        auto work = [&](int worker)
        {
            try
            {
                for (;;)
                {
                    const int i = next.fetch_add(1);
                    if (i >= n)
                        break;
                    fn(i, worker);
                }
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error)
                    error = std::current_exception();
                next.store(n);
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(threads - 1);
        for (int t = 1; t < threads; ++t)
            pool.emplace_back(work, t);
        work(0);
        for (auto &t : pool)
            t.join();
        if (error)
            std::rethrow_exception(error);
    }

    inline int hardware_threads()
    {
        const unsigned n = std::thread::hardware_concurrency();
        return n == 0 ? 1 : static_cast<int>(n);
    }

} // namespace pw
