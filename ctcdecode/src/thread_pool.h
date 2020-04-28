#include <future>
#include <queue>
#include <thread>
#include <vector>

class thread_pool
{
private:
    typedef std::packaged_task<void(size_t)> task_type;

    std::vector<std::thread> workers;
    std::queue<task_type> tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    thread_pool(size_t threads)
    {
        for (size_t i = 0; i < threads; i++)
        {
            this->workers.emplace_back([this, i] {
                for (;;)
                {
                    task_type task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || this->tasks.size() > 0; });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task(i);
                }
            });
        }
    }

    ~thread_pool()
    {
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->stop = true;
        }
        this->condition.notify_all();
        for (auto& worker : this->workers)
            worker.join();
    }

    size_t num_threads() const
    {
        return this->workers.size();
    }

    template <typename F>
    std::future<void> enqueue(F&& f)
    {
        auto task = task_type { f };

        std::future<void> res = task.get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if (this->stop)
                throw std::runtime_error("enqueue on stopped thread_pool");

            this->tasks.push(std::move(task));
        }
        this->condition.notify_one();

        return res;
    }

    template <typename F>
    void parallel_for(size_t start, size_t end, F&& f)
    {
        if (end - start < 2)
        {
            for (size_t i = start; i < end; i++)
                f(i, 0);
            return;
        }

        std::vector<std::future<void>> futures;
        for (size_t i = start; i < end; i++)
        {
            futures.push_back(this->enqueue([i, &f](size_t thread_idx) { f(i, thread_idx); }));
        }
        for (auto& future : futures)
            future.wait();
    }
};
