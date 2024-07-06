#include <iostream>
#include <thread>
#include <functional>
#include <atomic>
#include <memory>
#include <vector>
#include <openssl/evp.h>
#include <sstream>
#include <queue> // 使用标准库优先队列  
#include <mutex> // 使用互斥锁
#include <condition_variable>


using namespace std;
// 加密解密密钥
const string encryption_key = "66"; // 替换为您的加密密钥

// 任务结构体，包含任务执行函数和估计的执行时间
struct Task {
    function<void()> function;
    int estimated_execution_time; // 任务的估计执行时间

    // 比较函数，用于任务的排序
    bool operator<(const Task& other) const {
        // 按照估计的执行时间进行排序，执行时间越短的任务越优先
        return estimated_execution_time > other.estimated_execution_time;
    }
};

// 线程安全的任务队列的实现
class ThreadSafeQueue {
private:
    //mutable 关键字的作用是允许在常量成员函数中修改被声明为 mutable 的成员变量。
    //互斥锁 m 会被用来保护对优先队列 data_queue 的读写操作，确保在多线程环境下的线程安全性

    priority_queue<Task> data_queue; // 优先队列，按照任务的估计执行时间排序
    mutable mutex m; // 互斥锁

public:
    // 入队操作
    void push(const Task& task) {
        lock_guard<mutex> lock(m);
        data_queue.push(task);
    }

    // 出队操作
    Task pop() {
        lock_guard<mutex> lock(m);
        if (data_queue.empty()) {
            return Task{ [] {}, 0 }; // 返回空任务
        }
        Task res = data_queue.top();
        data_queue.pop();
        return res;
    }

    // 判断队列是否为空
    bool empty() const {
        lock_guard<mutex> lock(m);
        return data_queue.empty();
    }
};

// 线程池类
class ThreadPool {
public:
    // 构造函数，接受线程池的初始线程数量
    ThreadPool(size_t numThreads) : stop(false) {
        // 创建指定数量的线程，并将它们添加到线程池中
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this] {
                // 线程循环执行任务
                while (true) {
                    Task task;
                    {
                        unique_lock<std::mutex> lock(queue_mutex);
                        // 等待队列不为空或者线程池不停止
                        condition.wait(lock, [this] { return !taskQueue.empty() || stop; });
                        // 如果线程池停止并且队列为空，则退出循环
                        if (stop && taskQueue.empty()) {
                            return;
                        }
                        // 从队列中获取任务
                        task = taskQueue.pop();
                    }
                    // 执行任务
                    task.function();
                }
                });
        }
        cout << "线程池创建，包含 " << numThreads << " 个线程。\n";
    }

    // 析构函数，用于线程池的清理工作
    ~ThreadPool() {
        {
            lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all(); // 通知所有线程停止
        for (auto& thread : threads)
            thread.join();
    }

    // 添加任务到线程池的任务队列中
    void enqueue(Task task) {
        {
            lock_guard<std::mutex> lock(queue_mutex);
            taskQueue.push(task);
        }
        condition.notify_one(); // 通知一个线程有新任务
    }

    // 新增方法：动态调整线程池大小
    void resize(size_t numThreads) {
        if (numThreads == 0) {
            // 如果目标线程数为0，则不做任何调整
            return;
        }

        // 获取当前线程池中的线程数量
        size_t currentSize = threads.size();

        if (numThreads == currentSize) {
            // 如果目标线程数与当前线程数相同，则无需调整
            cout << "线程池大小已是 " << numThreads << "，无需调整。\n";
            return;
        }

        if (numThreads > currentSize) {
            // 增加线程数量
            size_t diff = numThreads - currentSize; // 计算需要增加的线程数

            // 向线程池中添加新线程
            for (size_t i = 0; i < diff; ++i) {
                threads.emplace_back([this] {
                    // 每个线程执行的任务
                    while (true) {
                        Task task;
                        {
                            unique_lock<std::mutex> lock(queue_mutex);
                            // 等待任务队列不为空或者线程池停止
                            condition.wait(lock, [this] { return !taskQueue.empty() || stop; });
                            if (stop && taskQueue.empty()) {
                                // 如果线程池停止且任务队列为空，则退出循环
                                return;
                            }
                            // 从任务队列中取出一个任务
                            task = taskQueue.pop();
                        }
                        // 执行任务
                        task.function();
                    }
                    });
            }
            // 输出增加后的线程池大小
            cout << "增加线程池大小到 " << numThreads << " 个线程。\n";
        }
        else {
            // 减少线程数量
            stop = true; // 设置停止标志，通知所有线程停止
            condition.notify_all(); // 唤醒所有线程

            // 等待所有线程完成执行
            for (auto& thread : threads) {
                thread.join();
            }

            // 清空线程容器
            threads.clear();

            stop = false; // 重新启动线程池

            // 重新创建指定数量的线程
            for (size_t i = 0; i < numThreads; ++i) {
                threads.emplace_back([this] {
                    // 每个线程执行的任务
                    while (true) {
                        Task task;
                        {
                            unique_lock<std::mutex> lock(queue_mutex);
                            // 等待任务队列不为空或者线程池停止
                            condition.wait(lock, [this] { return !taskQueue.empty() || stop; });
                            if (stop && taskQueue.empty()) {
                                // 如果线程池停止且任务队列为空，则退出循环
                                return;
                            }
                            // 从任务队列中取出一个任务
                            task = taskQueue.pop();
                        }
                        // 执行任务
                        task.function();
                    }
                    });
            }
            // 输出减少后的线程池大小
            cout << "减少线程池大小到 " << numThreads << " 个线程。\n";
        }
    }



private:
    vector<thread> threads; // 存储线程的容器
    ThreadSafeQueue taskQueue; // 存储任务的线程安全队列
    atomic<bool> stop; // 标志，表示线程池是否停止
    mutex queue_mutex; // 互斥锁，保护任务队列
    condition_variable condition; // 条件变量，用于线程间通信
};


// 加密函数
std::string encrypt(const std::string& plaintext) {
    EVP_CIPHER_CTX* ctx;
    unsigned char iv[EVP_MAX_IV_LENGTH] = { 0 }; // 初始化向量，这里设置为0
    unsigned char key[EVP_MAX_KEY_LENGTH];
    copy(encryption_key.begin(), encryption_key.end(), key);
    unsigned char ciphertext[1024] = { 0 }; // 加密后的数据缓冲区
    int len;
    int ciphertext_len;

    // 创建并初始化加密上下文
    if (!(ctx = EVP_CIPHER_CTX_new())) {
        cerr << "创建加密上下文时出错" << endl;
        return "";
    }

    // 初始化加密操作，选择加密算法和密钥
    if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {
        cerr <<  "初始化加密操作时出错" << endl;
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }

    // 执行加密操作
    if (1 != EVP_EncryptUpdate(ctx, ciphertext, &len, reinterpret_cast<const unsigned char*>(plaintext.c_str()), plaintext.length())) {
        cerr << "结束加密操作时出错" << endl;
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    ciphertext_len = len;

    // 结束加密操作，处理可能剩余的数据
    if (1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) {
        cerr << "Error finalizing encryption" << endl;
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    ciphertext_len += len;

    // 释放加密上下文并返回加密后的数据
    EVP_CIPHER_CTX_free(ctx);
    return string(reinterpret_cast<char*>(ciphertext), ciphertext_len);
}

// 解密函数
std::string decrypt(const std::string& ciphertext) {
    EVP_CIPHER_CTX* ctx;
    unsigned char iv[EVP_MAX_IV_LENGTH] = { 0 }; // 初始化向量，这里设置为0
    //确保在加密后的数据存储之前，缓冲区中没有任何未定义的数据

    unsigned char key[EVP_MAX_KEY_LENGTH];
    copy(encryption_key.begin(), encryption_key.end(), key);
    unsigned char plaintext[1024] = { 0 }; // 解密后的数据缓冲区
    int len;
    int plaintext_len;

    // 创建并初始化解密上下文
    if (!(ctx = EVP_CIPHER_CTX_new())) {
        cerr << "创建解密上下文时出错" << endl;
        return "";
    }

    // 初始化解密操作，选择加密算法和密钥
    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {
        cerr << "初始化解密操作时出错" << endl;
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }

    // 执行解密操作
    if (1 != EVP_DecryptUpdate(ctx, plaintext, &len, reinterpret_cast<const unsigned char*>(ciphertext.c_str()), ciphertext.length())) {
        cerr << "执行解密操作时出错"  << endl;
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    plaintext_len = len;

    // 结束解密操作，处理可能剩余的数据
    if (1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) {
        cerr << "结束解密操作时出错"  << endl;
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    plaintext_len += len;

    // 释放解密上下文并返回解密后的数据
    EVP_CIPHER_CTX_free(ctx);
    return string(reinterpret_cast<char*>(plaintext), plaintext_len);
}

int main() {
    // 创建一个包含4个线程的线程池
    ThreadPool pool(4);

    // 向线程池中添加10个任务
    for (int i = 0; i < 10; ++i) {
        // 使用lambda表达式作为任务，并输出任务执行情况
        Task task;
        task.function = [i] {
            stringstream ss;
            ss << "任务 " << i << " 的数据"; // 使用 stringstream 构造字符串
            string plaintext = ss.str();
            string encrypted_data = encrypt(plaintext);
            cout << "加密后的数据：" << encrypted_data << "\n";
            // 模拟解密过程
            string decrypted_data = decrypt(encrypted_data);
            cout << "解密后的数据：" << decrypted_data << "\n";
            };
        // 设置任务的估计执行时间
        task.estimated_execution_time = 1; // 这里假设所有任务的执行时间相同
        pool.enqueue(task);
    }

    // 假设需要在一定时间后调整线程池大小
    this_thread::sleep_for(chrono::seconds(1));

    // 动态调整线程池大小为 8a
    pool.resize(8);

    return 0;
}
//C++ 实现一个具有多线程任务管理功能的系统，并集成了简单的加密和解密功能
//这种系统可以用于需要高并发处理任务的场景，例如网络服务器、数据处理系统等。