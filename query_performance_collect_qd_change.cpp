#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
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

map<int, vector<float>>
test_vs_recall_char(
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    L2SpaceI &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {

        vector<size_t> efs;
        map<int, vector<float>> results;

        for (int i = k; i < 50; i=i+5) {
            efs.push_back(i);
        }
        for (int i = 50; i < 100; i += 10) {
            efs.push_back(i);
        }
        for (int i = 100; i < 300; i += 20) {
            efs.push_back(i);
        }
        for (int i = 300; i < 700; i += 20) {
            efs.push_back(i);
        }
        for (int i = 700; i < 1000; i += 30) {
            efs.push_back(i);
        }
        for (int i = 1000; i < 2000; i += 100) {
            efs.push_back(i);
        }
        for (int i = 2000; i <= 5000; i += 100) {
            efs.push_back(i);
        }

        for (size_t ef : efs) {
            appr_alg.setEf(ef);
            hnswlib::resetGlobalCallCount();

            StopW stopw = StopW();  // Start the stopwatch

            float recall = test_approx_char(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
            float time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
            size_t count = hnswlib::getGlobalCallCount();

            results[ef] = {recall, count, time_us_per_query};

            if (ef >= 300 && (recall >= 0.995 || ef == 5000)) {
                break;
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

map<int, vector<float>>
test_vs_recall_float(
    float *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    L2Space &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {

        vector<size_t> efs;
        map<int, vector<float>> results;

        for (int i = k; i < 50; i=i+5) {
            efs.push_back(i);
        }
        for (int i = 50; i < 100; i += 10) {
            efs.push_back(i);
        }
        for (int i = 100; i < 300; i += 20) {
            efs.push_back(i);
        }
        for (int i = 300; i < 700; i += 20) {
            efs.push_back(i);
        }
        for (int i = 700; i < 1000; i += 30) {
            efs.push_back(i);
        }
        for (int i = 1000; i < 2000; i += 100) {
            efs.push_back(i);
        }
        for (int i = 2000; i <= 5000; i += 100) {
            efs.push_back(i);
        }

        for (size_t ef : efs) {
            appr_alg.setEf(ef);
            hnswlib::resetGlobalCallCount();

            StopW stopw = StopW();  // Start the stopwatch

            float recall = test_approx_float(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
            float time_us_per_query = 1e-6 * stopw.getElapsedTimeMicro() / qsize;
            size_t count = hnswlib::getGlobalCallCount();

            results[ef] = {recall, count, time_us_per_query};

             if (ef >= 300 && (recall >= 0.995 || ef == 5000)) {
                break;
            }

        }
        return results;
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


int main(int argc, char* argv[]) {
    if (argc < 18) {
        cerr << "Usage: " << argv[0] << " filename base_path query_path indice_path index_path index_csv subdir size dim ef_construction m" << endl;
        return -1;
    }

    const string filename = argv[1];
    const char* path_b = argv[2];
    const char* path_q1 = argv[3];
    const char* path_q2 = argv[4];
    const char* path_q3 = argv[5];
    const char* path_q4 = argv[6];
    const char* path_i1 = argv[7];
    const char* path_i2 = argv[8];
    const char* path_i3 = argv[9];
    const char* path_i4 = argv[10];
    const char* path_index = argv[11];
    const char* path_csv = argv[12];
    const string subdir = argv[13];
    size_t vecsize = atoi(argv[14]);
    size_t vecdim= atoi(argv[15]);
    int efConstruction = atoi(argv[16]);
    int M = atoi(argv[17]);

    std::string str1 = "_25";
    std::string str2 = "_50";
    std::string str3 = "_75";
    std::string str4 = "_100";

    std::string filename1 = filename + str1;
    std::string filename2 = filename + str2;
    std::string filename3 = filename + str3;
    std::string filename4 = filename + str4;

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

    cout << "Loading 4 GT:\n";
    ifstream inputI1(path_i1, ios::binary);
    unsigned int *massQA1 = new unsigned int[qsize * 100];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputI1.read((char *) &t, 4);
        inputI1.read((char *) (massQA1 + 100 * i), t * 4);
        if (t != 100) {
            cout << "err";
            return -1;
        }
    }
    inputI1.close();

    ifstream inputI2(path_i2, ios::binary);
    unsigned int *massQA2 = new unsigned int[qsize * 100];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputI2.read((char *) &t, 4);
        inputI2.read((char *) (massQA2 + 100 * i), t * 4);
        if (t != 100) {
            cout << "err";
            return -1;
        }
    }
    inputI2.close();

    ifstream inputI3(path_i3, ios::binary);
    unsigned int *massQA3 = new unsigned int[qsize * 100];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputI3.read((char *) &t, 4);
        inputI3.read((char *) (massQA3 + 100 * i), t * 4);
        if (t != 100) {
            cout << "err";
            return -1;
        }
    }
    inputI3.close();

    ifstream inputI4(path_i4, ios::binary);
    unsigned int *massQA4 = new unsigned int[qsize * 100];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputI4.read((char *) &t, 4);
        inputI4.read((char *) (massQA4 + 100 * i), t * 4);
        if (t != 100) {
            cout << "err";
            return -1;
        }
    }
    inputI4.close();

    if (subdir == "sift") {
        unsigned char *massb = new unsigned char[vecdim];

        unsigned char *massQ1 = new unsigned char[qsize * vecdim];
        unsigned char *massQ2 = new unsigned char[qsize * vecdim];
        unsigned char *massQ3 = new unsigned char[qsize * vecdim];
        unsigned char *massQ4 = new unsigned char[qsize * vecdim];

        cout << "Loading 4 queries:\n";
        ifstream inputQ1(path_q1, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ1.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ1.close();
                return -1;
            }
           inputQ1.read((char *) massb, in);
            for (int j = 0; j < vecdim; j++) {
                massQ1[i * vecdim + j] = massb[j];
            }
        }
        inputQ1.close();

        ifstream inputQ2(path_q2, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ2.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ2.close();
                return -1;
            }
           inputQ2.read((char *) massb, in);
            for (int j = 0; j < vecdim; j++) {
                massQ2[i * vecdim + j] = massb[j];
            }
        }
        inputQ2.close();

        ifstream inputQ3(path_q3, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ3.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ3.close();
                return -1;
            }
           inputQ3.read((char *) massb, in);
            for (int j = 0; j < vecdim; j++) {
                massQ3[i * vecdim + j] = massb[j];
            }
        }
        inputQ3.close();

        ifstream inputQ4(path_q4, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ4.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ4.close();
                return -1;
            }
           inputQ4.read((char *) massb, in);
            for (int j = 0; j < vecdim; j++) {
                massQ4[i * vecdim + j] = massb[j];
            }
        }
        inputQ4.close();

        L2SpaceI l2space(vecdim);
        hnswlib::resetGlobalCallCount();

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

            if (build_time > 54000){
                    appr_alg->saveIndex(path_index);
                }
        }

        size_t c_dc_counts = hnswlib::getGlobalCallCount();

        cout << "Calculating recall rate:\n";

        size_t k = 10;

        vector<std::priority_queue<std::pair<int, labeltype >>> answers1;
        get_gt_char(massQA1, massQ1, vecsize, qsize, l2space, vecdim, answers1, k);
        std::map<int, vector<float>> all_results1 = test_vs_recall_char(massQ1, vecsize, qsize, *appr_alg, l2space, vecdim, answers1, k);

        vector<std::priority_queue<std::pair<int, labeltype >>> answers2;
        get_gt_char(massQA2, massQ2, vecsize, qsize, l2space, vecdim, answers2, k);
        std::map<int, vector<float>> all_results2 = test_vs_recall_char(massQ2, vecsize, qsize, *appr_alg, l2space, vecdim, answers2, k);

        vector<std::priority_queue<std::pair<int, labeltype >>> answers3;
        get_gt_char(massQA3, massQ3, vecsize, qsize, l2space, vecdim, answers3, k);
        std::map<int, vector<float>> all_results3 = test_vs_recall_char(massQ3, vecsize, qsize, *appr_alg, l2space, vecdim, answers3, k);

        vector<std::priority_queue<std::pair<int, labeltype >>> answers4;
        get_gt_char(massQA4, massQ4, vecsize, qsize, l2space, vecdim, answers4, k);
        std::map<int, vector<float>> all_results4 = test_vs_recall_char(massQ4, vecsize, qsize, *appr_alg, l2space, vecdim, answers4, k);

        std::ofstream csvFile(path_csv, std::ios::app);
        for (const auto& pair : all_results1) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename1 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        for (const auto& pair : all_results2) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename2 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        for (const auto& pair : all_results3) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename3 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        for (const auto& pair : all_results4) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename4 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        csvFile.close();

        delete[] massQA1;
        delete[] massQA2;
        delete[] massQA3;
        delete[] massQA4;
        delete[] massQ1;
        delete[] massQ2;
        delete[] massQ3;
        delete[] massQ4;
        delete appr_alg;
    }
    else{
        float *massb = new float[vecdim];

        float *massQ1 = new float[qsize * vecdim];
        float *massQ2 = new float[qsize * vecdim];
        float *massQ3 = new float[qsize * vecdim];
        float *massQ4 = new float[qsize * vecdim];

        cout << "Loading 4 queries:\n";
        ifstream inputQ1(path_q1, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ1.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ1.close();
                return -1;
            }
           inputQ1.read((char *) massb, in * sizeof(float));
            for (int j = 0; j < vecdim; j++) {
                massQ1[i * vecdim + j] = massb[j];
            }
        }
        inputQ1.close();

        ifstream inputQ2(path_q2, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ2.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ2.close();
                return -1;
            }
           inputQ2.read((char *) massb, in * sizeof(float));
            for (int j = 0; j < vecdim; j++) {
                massQ2[i * vecdim + j] = massb[j];
            }
        }
        inputQ2.close();

        ifstream inputQ3(path_q3, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ3.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ3.close();
                return -1;
            }
           inputQ3.read((char *) massb, in * sizeof(float));
            for (int j = 0; j < vecdim; j++) {
                massQ3[i * vecdim + j] = massb[j];
            }
        }
        inputQ3.close();

        ifstream inputQ4(path_q4, ios::binary);
        for (int i = 0; i < qsize; i++) {
            int in = 0;
            inputQ4.read((char *)&in, 4);

            if (in != vecdim) {
                cerr << "file error." << endl;
                inputQ4.close();
                return -1;
            }
           inputQ4.read((char *) massb, in * sizeof(float));
            for (int j = 0; j < vecdim; j++) {
                massQ4[i * vecdim + j] = massb[j];
            }
        }
        inputQ4.close();

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

            if (build_time > 30000){
                appr_alg->saveIndex(path_index);
            }
        }

        size_t c_dc_counts = hnswlib::getGlobalCallCount();

        cout << "Calculating recall rate:\n";

        size_t k = 10;

        vector<std::priority_queue<std::pair<float, labeltype >>> answers1;
        get_gt_float(massQA1, massQ1, vecsize, qsize, l2space, vecdim, answers1, k);
        std::map<int, vector<float>> all_results1 = test_vs_recall_float(massQ1, vecsize, qsize, *appr_alg, l2space, vecdim, answers1, k);

        vector<std::priority_queue<std::pair<float, labeltype >>> answers2;
        get_gt_float(massQA2, massQ2, vecsize, qsize, l2space, vecdim, answers2, k);
        std::map<int, vector<float>> all_results2 = test_vs_recall_float(massQ2, vecsize, qsize, *appr_alg, l2space, vecdim, answers2, k);

        vector<std::priority_queue<std::pair<float, labeltype >>> answers3;
        get_gt_float(massQA3, massQ3, vecsize, qsize, l2space, vecdim, answers3, k);
        std::map<int, vector<float>> all_results3 = test_vs_recall_float(massQ3, vecsize, qsize, *appr_alg, l2space, vecdim, answers3, k);

        vector<std::priority_queue<std::pair<float, labeltype >>> answers4;
        get_gt_float(massQA4, massQ4, vecsize, qsize, l2space, vecdim, answers4, k);
        std::map<int, vector<float>> all_results4 = test_vs_recall_float(massQ4, vecsize, qsize, *appr_alg, l2space, vecdim, answers4, k);

        std::ofstream csvFile(path_csv, std::ios::app);
        for (const auto& pair : all_results1) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename1 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        for (const auto& pair : all_results2) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename2 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        for (const auto& pair : all_results3) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename3 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        for (const auto& pair : all_results4) {
            int efS = pair.first;
            float rec = pair.second[0];
            size_t s_dc_counts = pair.second[1];
            float tpq = pair.second[2];

            csvFile << filename4 << "," << efConstruction << "," << M << "," << efS << "," << build_time << "," << memory_usage << "," << rec << "," << tpq << "," << c_dc_counts << "," << s_dc_counts << "\n";
        }
        csvFile.close();

        delete[] massQA1;
        delete[] massQA2;
        delete[] massQA3;
        delete[] massQA4;
        delete[] massQ1;
        delete[] massQ2;
        delete[] massQ3;
        delete[] massQ4;
        delete appr_alg;
    }

    return 0;
}
