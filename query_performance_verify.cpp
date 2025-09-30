#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <map>
#include "./hnswlib/hnswlib.h"
#include "./hnswlib/space_l2.h"

#include <unordered_set>

using namespace std;
using namespace hnswlib;

namespace hnswlib {
    std::atomic<size_t> globalCallCount(0);

    size_t getGlobalCallCount() {
        return globalCallCount.load();
    }

    void resetGlobalCallCount() {
        globalCallCount.store(0);
    }
}

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

//针对整数数据集
static void
get_gt_char(
    unsigned int *massQA,
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    L2SpaceI &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    (vector<std::priority_queue<std::pair<int, labeltype >>>(qsize)).swap(answers);
//    DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[100 * i + j]);
        }
    }
}

static float
test_approx_char(
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    size_t correct = 0;
    size_t total = 0;
    // uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        std::priority_queue<std::pair<int, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<int, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

vector<std::tuple<int, float, float, size_t, float>>
test_vs_recall_char(
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    L2SpaceI &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k,
    int pr_efS,
    float target_rec) {

        int dealt = 0;
        std::vector<std::tuple<int, float, float, size_t, float>> results;

        if (pr_efS < 100){
            dealt = 1;
        }
        else if (100 <= pr_efS && pr_efS < 300){
            dealt = 5;
        }
        else if (300 <= pr_efS && pr_efS < 500){
            dealt = 10;
        }
        else if (500 <= pr_efS && pr_efS < 900){
            dealt = 20;
        }
        else if (900 <= pr_efS && pr_efS < 1800){
            dealt = 50;
        }
        else if (1800 <= pr_efS && pr_efS < 3000){
            dealt = 100;
        }
        else {
            dealt = 200;
        }

        appr_alg.setEf(pr_efS);
        hnswlib::resetGlobalCallCount();

        StopW stopw_global = StopW();  // Start the stopwatch
        StopW stopw = StopW();

        float recall = test_approx_char(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
        size_t count = hnswlib::getGlobalCallCount();

        if (recall >= target_rec){
            results.emplace_back(std::make_tuple(pr_efS, recall, time_us_per_query, count, time_us_per_query * qsize));

            for (int ef = pr_efS - dealt; ef >= 10; ef = ef - dealt) {
                if (ef < 100){
                    dealt = 1;
                }
                else if (100 <= ef && ef < 300){
                    dealt = 5;
                }
                else if (300 <= ef && ef < 500){
                    dealt = 10;
                }
                else if (500 <= ef && ef < 900){
                    dealt = 20;
                }
                else if (900 <= ef && ef < 1800){
                    dealt = 50;
                }
                else if (1800 <= ef && ef < 3000){
                    dealt = 100;
                }
                else {
                    dealt = 200;
                }

                appr_alg.setEf(ef);
                hnswlib::resetGlobalCallCount();

                StopW stopw = StopW();  // Start the stopwatch

                float new_recall = test_approx_char(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
                float new_time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
                float paras_search_time = 1e-6 * stopw_global.getElapsedTimeMicro();
                size_t new_count = hnswlib::getGlobalCallCount();

                if (new_recall >= target_rec) {
                    results.clear();
                    results.emplace_back(std::make_tuple(ef, new_recall, new_time_us_per_query, new_count, paras_search_time));
                }
                else{
                    break;
                }
            }
        }
        else {
            for (int ef = pr_efS + dealt; ef <= 8000; ef = ef + dealt) {
                if (ef < 100){
                    dealt = 1;
                }
                else if (100 <= ef && ef < 300){
                    dealt = 5;
                }
                else if (300 <= ef && ef < 500){
                    dealt = 10;
                }
                else if (500 <= ef && ef < 900){
                    dealt = 20;
                }
                else if (900 <= ef && ef < 1800){
                    dealt = 50;
                }
                else if (1800 <= ef && ef < 3000){
                    dealt = 100;
                }
                else {
                    dealt = 200;
                }

                appr_alg.setEf(ef);
                hnswlib::resetGlobalCallCount();

                StopW stopw = StopW();  // Start the stopwatch

                float new_recall = test_approx_char(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
                float new_time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
                float paras_search_time = 1e-6 * stopw_global.getElapsedTimeMicro();
                size_t new_count = hnswlib::getGlobalCallCount();

                if (new_recall >= target_rec) {
                    results.emplace_back(std::make_tuple(ef, new_recall, new_time_us_per_query, new_count, paras_search_time));
                    break;
                }
                else if (ef >= 8000) {
                    results.emplace_back(std::make_tuple(ef, new_recall, new_time_us_per_query, new_count, paras_search_time));
                    break;
                }
            }
        }

        return results;
}

static void
get_gt_float(
    unsigned int *massQA,
    float *massQ,

    size_t vecsize,
    size_t qsize,
    L2Space &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
//    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[100 * i + j]);
        }
    }
}

static float
test_approx_float(
    float *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    size_t correct = 0;
    size_t total = 0;
    // uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

vector<std::tuple<int, float, float, size_t, float>>
test_vs_recall_float(
    float *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    L2Space &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k,
    int pr_efS,
    float target_rec) {

        int dealt = 0;
        std::vector<std::tuple<int, float, float, size_t, float>> results;

        if (pr_efS < 100){
            dealt = 1;
        }
        else if (100 <= pr_efS && pr_efS < 300){
            dealt = 5;
        }
        else if (300 <= pr_efS && pr_efS < 500){
            dealt = 10;
        }
        else if (500 <= pr_efS && pr_efS < 900){
            dealt = 20;
        }
        else if (900 <= pr_efS && pr_efS < 1800){
            dealt = 50;
        }
        else if (1800 <= pr_efS && pr_efS < 3000){
            dealt = 100;
        }
        else {
            dealt = 200;
        }

        appr_alg.setEf(pr_efS);
        hnswlib::resetGlobalCallCount();

        StopW stopw_global = StopW();  // Start the stopwatch
        StopW stopw = StopW();

        float recall = test_approx_float(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
        size_t count = hnswlib::getGlobalCallCount();

        if (recall >= target_rec){
            results.emplace_back(std::make_tuple(pr_efS, recall, time_us_per_query, count, time_us_per_query * qsize));

            for (int ef = pr_efS - dealt; ef >= 10; ef = ef - dealt) {
                if (ef < 100){
                    dealt = 1;
                }
                else if (100 <= ef && ef < 300){
                    dealt = 5;
                }
                else if (300 <= ef && ef < 500){
                    dealt = 10;
                }
                else if (500 <= ef && ef < 900){
                    dealt = 20;
                }
                else if (900 <= ef && ef < 1800){
                    dealt = 50;
                }
                else if (1800 <= ef && ef < 3000){
                    dealt = 100;
                }
                else {
                    dealt = 200;
                }

                appr_alg.setEf(ef);
                hnswlib::resetGlobalCallCount();

                StopW stopw = StopW();  // Start the stopwatch

                float new_recall = test_approx_float(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
                float new_time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
                float paras_search_time = 1e-6 * stopw_global.getElapsedTimeMicro();
                size_t new_count = hnswlib::getGlobalCallCount();

                if (new_recall >= target_rec) {
                    results.clear();
                    results.emplace_back(std::make_tuple(ef, new_recall, new_time_us_per_query, new_count, paras_search_time));
                }
                else{
                    break;
                }
            }
        }
        else {
            for (int ef = pr_efS + dealt; ef <= 8000; ef = ef + dealt) {
                if (ef < 100){
                    dealt = 1;
                }
                else if (100 <= ef && ef < 300){
                    dealt = 5;
                }
                else if (300 <= ef && ef < 500){
                    dealt = 10;
                }
                else if (500 <= ef && ef < 900){
                    dealt = 20;
                }
                else if (900 <= ef && ef < 1800){
                    dealt = 50;
                }
                else if (1800 <= ef && ef < 3000){
                    dealt = 100;
                }
                else {
                    dealt = 200;
                }

                appr_alg.setEf(ef);
                hnswlib::resetGlobalCallCount();

                StopW stopw = StopW();  // Start the stopwatch

                float new_recall = test_approx_float(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
                float new_time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
                float paras_search_time = 1e-6 * stopw_global.getElapsedTimeMicro();
                size_t new_count = hnswlib::getGlobalCallCount();

                if (new_recall >= target_rec) {
                    results.emplace_back(std::make_tuple(ef, new_recall, new_time_us_per_query, new_count, paras_search_time));
                    break;
                }
                else if (ef >= 8000) {
                    results.emplace_back(std::make_tuple(ef, new_recall, new_time_us_per_query, new_count, paras_search_time));
                    break;
                }
            }
        }
        
        return results;
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


int main(int argc, char* argv[]) {
    if (argc < 14) {
        cerr << "Usage: " << argv[0] << " filename base_path query_path indice_path index_path index_csv subdir size dim ef_construction m" << endl;
        return -1;
    }

    const string filename = argv[1];
    const char* path_b = argv[2];
    const char* path_q = argv[3];
    const char* path_i = argv[4];
    const char* path_index = argv[5];
    const char* path_csv = argv[6];
    const string subdir = argv[7];
    size_t vecsize = atoi(argv[8]);
    size_t vecdim= atoi(argv[9]);
    int efConstruction = atoi(argv[10]);
    int M = atoi(argv[11]);
    int pr_efS = atoi(argv[12]);
    float target_rec = std::stof(argv[13]);

    int qsize = 10000;
    if (subdir == "msong") {
            qsize = 200;
        }
    else if (subdir == "gist"){
        qsize = 1000;
    }
    else{
        qsize = 10000;
    }

    float memory_usage = 0;
    float build_time = 0;

    cout << "Loading GT:\n";
    ifstream inputI(path_i, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 100];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputI.read((char *) &t, 4);
        inputI.read((char *) (massQA + 100 * i), t * 4);
        if (t != 100) {
            cout << "err";
            return -1;
        }
    }

    inputI.close();

    if (subdir == "sift") {
        cout << "Loading queries:\n";
        unsigned char *massb = new unsigned char[vecdim];
        unsigned char *massQ = new unsigned char[qsize * vecdim];

        ifstream inputQ(path_q, ios::binary);

        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ.close();
                return -1;
            }
           inputQ.read((char *) massb, in);
            for (int j = 0; j < vecdim; j++) {
                massQ[i * vecdim + j] = massb[j];
            }
        }
        inputQ.close();

        L2SpaceI l2space(vecdim);
        hnswlib::resetGlobalCallCount();  // 明确重置距离函数调用次数

        HierarchicalNSW<int> *appr_alg;
        if (exists_test(path_index)) {
            cout << "Loading index from " << path_index << ":\n";
            appr_alg = new HierarchicalNSW<int>(&l2space, path_index, false);

        }
        else{
            cout << "Building index:\n";
            appr_alg = new HierarchicalNSW<int>(&l2space, vecsize, M, efConstruction);

            ifstream input(path_b, ios::binary);

            int in = 0;
            input.read((char *) &in, 4);
            if (in != vecdim) {
                cout << "file error";
                exit(1);
            }
            input.read((char *) massb, in);

            appr_alg->addPoint((void *) (massb), (size_t) 0);

            int j1 = 0;
            StopW stopw_full = StopW();

#pragma omp parallel for
            for (int i = 1; i < vecsize; i++) {
                unsigned char mass[vecdim];
                int j2 = 0;
#pragma omp critical
                {
                    input.read((char *) &in, 4);
                    if (in != vecdim) {
                        cout << "file error";
                        exit(1);
                    }
                    input.read((char *) massb, in);
                    for (int j = 0; j < vecdim; j++) {
                        mass[j] = massb[j];
                    }
                    j1++;
                    j2 = j1;
                }
                appr_alg->addPoint((void *) (mass), (size_t) j2);
            }
            input.close();

            build_time = 1e-6 * stopw_full.getElapsedTimeMicro();
            memory_usage = getCurrentRSS() / 1000000;
            stopw_full.reset();

            if (build_time > 72000){
                    appr_alg->saveIndex(path_index);
                }
        }

        size_t c_dc_counts = hnswlib::getGlobalCallCount();

        cout << "Calculating recall rate:\n";
        vector<std::priority_queue<std::pair<int, labeltype >>> answers;
        size_t k = 10;

        get_gt_char(massQA, massQ, vecsize, qsize, l2space, vecdim, answers, k);
        std::vector<std::tuple<int, float, float, size_t, float>> results = test_vs_recall_char(massQ, vecsize, qsize, *appr_alg, l2space, vecdim, answers, k, pr_efS, target_rec);

        int real_efS = std::get<0>(results[0]);
        float real_rec = std::get<1>(results[0]);
        float tpq = std::get<2>(results[0]);
        size_t s_dc_counts = std::get<3>(results[0]);
        float p_st = std::get<4>(results[0]);

        std::ofstream csvFile(path_csv, std::ios::app);
        csvFile << filename << "," << efConstruction << "," << M << "," << pr_efS << "," << real_efS << "," << build_time << "," << memory_usage << "," << target_rec << "," << real_rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "," << p_st << "\n";

        csvFile.close();

        delete[] massQA;
        delete[] massQ;
        delete appr_alg;
    }
    else{
        cout << "Loading queries:\n";
        float *massb = new float[vecdim];
        float *massQ = new float[qsize * vecdim];

        ifstream inputQ(path_q, ios::binary);

        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ.close();
                return -1;
            }
            inputQ.read((char *) massb, in * sizeof(float));
            for (int j = 0; j < vecdim; j++) {
                massQ[i * vecdim + j] = massb[j];
            }
        }
        inputQ.close();

        L2Space l2space(vecdim);
        hnswlib::resetGlobalCallCount();

        HierarchicalNSW<float> *appr_alg;
        if (exists_test(path_index)) {
            cout << "Loading index from " << path_index << ":\n";
            appr_alg = new HierarchicalNSW<float>(&l2space, path_index, false);
        }
        else{
            cout << "Building index:\n";
            appr_alg = new HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction);

            ifstream input(path_b, ios::binary);

            int in = 0;
            input.read((char *) &in, 4);

            if (in != vecdim) {
                cout << "file error";
                exit(1);
            }
            input.read((char *) massb, in * sizeof(float));

            appr_alg->addPoint((void *) (massb), (size_t) 0);

            int j1 = 0;

            StopW stopw_full = StopW();

#pragma omp parallel for
            for (int i = 1; i < vecsize; i++) {
                float mass[vecdim];
                int j2 = 0;
#pragma omp critical
                {
                    input.read((char *) &in, 4);
                    if (in != vecdim) {
                        cout << "file error";
                        exit(1);
                    }
                    input.read((char *) massb, in * sizeof(float));
                    for (int j = 0; j < vecdim; j++) {
                        mass[j] = massb[j];
                    }
                    j1++;
                    j2 = j1;

                }
                appr_alg->addPoint((void *) (mass), (size_t) j2);
            }
            input.close();

            build_time = 1e-6 * stopw_full.getElapsedTimeMicro();
            memory_usage = getCurrentRSS() / 1000000;
            stopw_full.reset();

            if (build_time > 72000){
                appr_alg->saveIndex(path_index);
            }
        }

        size_t c_dc_counts = hnswlib::getGlobalCallCount();

        cout << "Calculating recall rate:\n";
        vector<std::priority_queue<std::pair<float, labeltype >>> answers;
        size_t k = 10;

        get_gt_float(massQA, massQ, vecsize, qsize, l2space, vecdim, answers, k);
        std::vector<std::tuple<int, float, float, size_t, float>> results = test_vs_recall_float(massQ, vecsize, qsize, *appr_alg, l2space, vecdim, answers, k, pr_efS, target_rec);

        int real_efS = std::get<0>(results[0]);
        float real_rec = std::get<1>(results[0]);
        float tpq = std::get<2>(results[0]);
        size_t s_dc_counts = std::get<3>(results[0]);
        float p_st = std::get<4>(results[0]);

        std::ofstream csvFile(path_csv, std::ios::app);
        csvFile << filename << "," << efConstruction << "," << M << "," << pr_efS << "," << real_efS << "," << build_time << "," << memory_usage << "," << target_rec << "," << real_rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "," << p_st << "\n";

        csvFile.close();

        delete[] massQA;
        delete[] massQ;
        delete appr_alg;
    }

    return 0;
}
