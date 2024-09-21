#include <iostream>
#include <thread>
#include <functional>
#include <atomic>
#include <memory>
#include <vector>
#include <openssl/evp.h>
#include <sstream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <fstream>
#include <chrono>

using namespace std;
mutex logger_mutex; // 用于保护 Logger 的访问
mutex encryptor_mutex; // 用于保护 Encryptor 的访问

// 日志记录类
class Logger {
private:
    ofstream log_file;

public:
    Logger(const string& file_name) {
        log_file.open(file_name, ios::out | ios::app);
        if (!log_file.is_open()) {
            throw runtime_error("无法打开日志文件");
        }
    }

    ~Logger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    void log(const string& message) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << chrono::system_clock::to_time_t(chrono::system_clock::now()) << "] " << message << endl;
    }

private:
    mutex log_mutex;
};

// Base64 编码字符集
static const string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

// Base64 编码函数
std::string base64_encode(const std::string& in) {
    string out;
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

// Base64 解码函数
std::string base64_decode(const std::string& in) {
    vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[base64_chars[i]] = i;

    string out;
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// 加密解密类
class Encryptor {
private:
    string key;
public:
    Encryptor(const string& encryption_key) : key(encryption_key) {}

    std::string encrypt(const std::string& plaintext) {
        EVP_CIPHER_CTX* ctx;
        unsigned char iv[EVP_MAX_IV_LENGTH] = { 0 };
        unsigned char key[EVP_MAX_KEY_LENGTH];
        copy(this->key.begin(), this->key.end(), key);
        unsigned char ciphertext[1024] = { 0 };
        int len;
        int ciphertext_len;

        if (!(ctx = EVP_CIPHER_CTX_new())) {
            cerr << "创建加密上下文时出错" << endl;
            return "";
        }

        if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {
            cerr << "初始化加密操作时出错" << endl;
            EVP_CIPHER_CTX_free(ctx);
            return "";
        }

        if (1 != EVP_EncryptUpdate(ctx, ciphertext, &len, reinterpret_cast<const unsigned char*>(plaintext.c_str()), plaintext.length())) {
            cerr << "加密操作时出错" << endl;
            EVP_CIPHER_CTX_free(ctx);
            return "";
        }
        ciphertext_len = len;

        if (1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) {
            cerr << "结束加密操作时出错" << endl;
            EVP_CIPHER_CTX_free(ctx);
            return "";
        }
        ciphertext_len += len;

        EVP_CIPHER_CTX_free(ctx);
        return base64_encode(string(reinterpret_cast<char*>(ciphertext), ciphertext_len));
    }

    std::string decrypt(const std::string& ciphertext) {
        EVP_CIPHER_CTX* ctx;
        unsigned char iv[EVP_MAX_IV_LENGTH] = { 0 };
        unsigned char key[EVP_MAX_KEY_LENGTH];
        copy(this->key.begin(), this->key.end(), key);
        string decoded_ciphertext = base64_decode(ciphertext);
        unsigned char plaintext[1024] = { 0 };
        int len;
        int plaintext_len;

        if (!(ctx = EVP_CIPHER_CTX_new())) {
            cerr << "创建解密上下文时出错" << endl;
            return "";
        }

        if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {
            cerr << "初始化解密操作时出错" << endl;
            EVP_CIPHER_CTX_free(ctx);
            return "";
        }

        if (1 != EVP_DecryptUpdate(ctx, plaintext, &len, reinterpret_cast<const unsigned char*>(decoded_ciphertext.c_str()), decoded_ciphertext.length())) {
            cerr << "解密操作时出错" << endl;
            EVP_CIPHER_CTX_free(ctx);
            return "";
        }
        plaintext_len = len;

        if (1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) {
            cerr << "结束解密操作时出错" << endl;
            EVP_CIPHER_CTX_free(ctx);
            return "";
        }
        plaintext_len += len;

        EVP_CIPHER_CTX_free(ctx);
        return string(reinterpret_cast<char*>(plaintext), plaintext_len);
    }
};

// 任务结构体，包含任务执行函数和估计的执行时间
struct Task {
    function<void()> function;
    int estimated_execution_time;

    bool operator<(const Task& other) const {
        return estimated_execution_time > other.estimated_execution_time;
    }
};

// 线程安全的任务队列的实现
class ThreadSafeQueue {
private:
    priority_queue<Task> data_queue;
    mutable mutex m;
    condition_variable data_available;

public:
    void push(const Task& task) {
        {
            lock_guard<mutex> lock(m);
            data_queue.push(task);
        }
        data_available.notify_one();
    }//lock 的生命周期也结束，lock_guard 的析构函数被调用。

    Task pop() {
        unique_lock<mutex> lock(m);
        while (data_queue.empty()) {
            data_available.wait(lock);
        }
        Task res = data_queue.top();  // 获取优先级最高的任务
        data_queue.pop();              // 移除这个任务
        return res;
    }

    bool empty() const {
        lock_guard<mutex> lock(m);
        return data_queue.empty();
    }
    //pop() 允许你移除并处理优先队列中的最高优先级任务，
    //确保后续任务能够根据新的优先级顺序进行处理。
};

// 线程池类
class ThreadPool {
public:
    ThreadPool(size_t numThreads, Logger* logger) : stop(false), logger(logger) {
        resize(numThreads);
    }

    ~ThreadPool() {
        {
            lock_guard<mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    //可以加入互斥锁
    void enqueue(Task task) {
        taskQueue.push(task);
    }

    void resize(size_t numThreads) {
        size_t currentSize = threads.size();
        if (numThreads == currentSize) return;

        if (numThreads > currentSize) {
            size_t diff = numThreads - currentSize;
            for (size_t i = 0; i < diff; ++i) {
                threads.emplace_back([this] {
                    while (true) {
                        Task task;
                        {
                            unique_lock<mutex> lock(queue_mutex);
                            condition.wait(lock, [this] { return !taskQueue.empty() || stop; });
                            if (stop && taskQueue.empty()) return;
                            task = taskQueue.pop();
                        }
                        logger->log("执行任务");
                        task.function();
                    }
                    });
            }
            logger->log("增加线程池大小到 " + to_string(numThreads) + " 个线程");
        }
        else {
            {
                lock_guard<mutex> lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            threads.clear();
            stop = false;
            for (size_t i = 0; i < numThreads; ++i) {
                threads.emplace_back([this] {
                    while (true) {
                        Task task;
                        {
                            unique_lock<mutex> lock(queue_mutex);
                            condition.wait(lock, [this] { return !taskQueue.empty() || stop; });
                            if (stop && taskQueue.empty()) return;
                            task = taskQueue.pop();
                        }
                        logger->log("执行任务");
                        task.function();
                    }
                    });
            }
            logger->log("减少线程池大小到 " + to_string(numThreads) + " 个线程");
        }
    }

private:
    vector<thread> threads;
    ThreadSafeQueue taskQueue;
    atomic<bool> stop;
    mutex queue_mutex;
    condition_variable condition;
    Logger* logger;
};

// 监控线程池状态
void monitor(ThreadPool& pool, Logger& logger, std::atomic<bool>& stop_monitor) {
    while (!stop_monitor) {
        // 将睡眠时间切分成小段，以便快速响应 stop_monitor 的变化
        for (int i = 0; i < 5 && !stop_monitor; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // 如果在睡眠周期内 stop_monitor 变为 true，就跳出循环
        if (stop_monitor) {
            break;
        }

        // 记录线程池的状态
        logger.log("监控：线程池正在运行...");
    }
    logger.log("监控：线程池监控已停止。");
}
//
//监控线程池状态
//void monitor(ThreadPool& pool, Logger& logger, atomic<bool>& stop_monitor) {
//    while (!stop_monitor) {
//        this_thread::sleep_for(chrono::seconds(5));
//        logger.log("监控：线程池正在运行...");
//    }
//}

int main() {
    Logger logger("log.txt");
    ThreadPool pool(4, &logger);
    Encryptor encryptor("66123456789123456789123456789123");
    std::atomic<bool> stop_monitor(false);

    // 启动监控线程
    std::thread monitor_thread(monitor, std::ref(pool), std::ref(logger), std::ref(stop_monitor));

    for (int i = 0; i < 10; ++i) {
        Task task;
        task.function = [i, &encryptor, &logger] {
            try {
                std::stringstream ss;
                ss << "任务 " << i << " 的数据";
                std::string plaintext = ss.str();

                std::lock_guard<std::mutex> encryptor_lock(encryptor_mutex); // 加锁以确保线程安全
                std::string encrypted_data = encryptor.encrypt(plaintext);

                {
                    std::lock_guard<std::mutex> logger_lock(logger_mutex); // 加锁以确保线程安全
                    logger.log("加密后的数据：" + encrypted_data);
                }

                std::string decrypted_data = encryptor.decrypt(encrypted_data);

                {
                    std::lock_guard<std::mutex> logger_lock(logger_mutex); // 加锁以确保线程安全
                    logger.log("解密后的数据：" + decrypted_data);
                }
            }
            catch (const std::exception& e) {
                std::lock_guard<std::mutex> logger_lock(logger_mutex); // 加锁以确保线程安全
                logger.log("处理任务 " + std::to_string(i) + " 时发生错误：" + e.what());
            }
            };
        task.estimated_execution_time = 1;
        pool.enqueue(task);
    }

    // 动态调整线程池大小
    std::this_thread::sleep_for(std::chrono::seconds(1));
    pool.resize(6); // 可以在这里增加逻辑来根据任务情况决定是否减少线程数

    // 停止监控
    stop_monitor = true;
    monitor_thread.join();

  

    return 0;
}
