#include <chrono>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include "stdio.h"
#include "stdlib.h"


class MyTimer {
public:
    void start(const std::string& name) {
        time_points[name] = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        durations[name] += std::chrono::duration<double, std::milli>(end_time - time_points[name]).count();
        count[name] += 1;
    }

    double elapsedMilliseconds(const std::string& name, const std::string& mode = "average") const {
        if (mode == "average")
            return durations.at(name)/count.at(name);
        else 
            return durations.at(name);
    }

    void printAllTimes(
        int iteration,
        int world_size,
        int local_rank,
        const char* log_folder = nullptr
    ) const {
        char* prefix = new char[100];
		sprintf(prefix, "\nit=%d,ws:%d,rk=%d  -->\n", iteration, world_size, local_rank);
		char* filename = new char[100];
		sprintf(filename, "%s/time_ws=%d_rk=%d.log", log_folder, world_size, local_rank);

        std::vector<std::pair<std::string, double>> sortedTimes(durations.begin(), durations.end());
        std::sort(sortedTimes.begin(), sortedTimes.end(), 
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );
        std::cout << prefix << std::endl;
        for (const auto& pair : sortedTimes) {
            std::cout << pair.first << " time: " << elapsedMilliseconds(pair.first, "sum") << " ms" << std::endl;
        }
        //save in file
        FILE *fp;
        fp = fopen(filename, "a");
        fprintf(fp, "%s", prefix);
        for (const auto& pair : sortedTimes) {
            fprintf(fp, "%s time: %f ms\n", pair.first.c_str(), elapsedMilliseconds(pair.first, "sum"));
        }
        //clean up
        fclose(fp);
        delete[] prefix;
        delete[] filename;
    }

private:
    std::unordered_map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> time_points;
    std::unordered_map<std::string, double> durations;
    std::unordered_map<std::string, int> count;
};