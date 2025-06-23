#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <filesystem>
#include <cfloat> 
#include <iomanip>

namespace fs = std::filesystem;

const int MAX_ITER = 100;
const float EPS = 1e-4f;

std::vector<float> load_features(const std::string& fname, int& n, int& dim) {
    std::ifstream in(fname);
    std::vector<float> v;
    float x;
    while (in >> x) v.push_back(x);
    n = static_cast<int>(v.size());
    dim = 1;
    return v;
}

std::vector<std::string> collect_feature_files(
    const std::string& base_dir,
    const std::vector<std::string>& classes
) {
    std::vector<std::string> files;
    for (const auto& cls : classes) {
        fs::path class_path = fs::path(base_dir) / cls;
        if (!fs::exists(class_path) || !fs::is_directory(class_path)) continue;
        for (auto& entry : fs::directory_iterator(class_path)) {
            if (!entry.is_regular_file()) continue;
            auto name = entry.path().filename().string();
            // Procurar pelo nome exato do arquivo gerado pelo OpenMP
            if (name.find("haralick_raw_omp.txt") != std::string::npos) {
                files.push_back(entry.path().string());
            }
        }
    }
    return files;
}

__global__ void assign_clusters_multiK(
    const float* data, int n, int dim,
    const float* centroids, const int* ks, const int* offsets,
    int num_ks, int* labels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float* pt = &data[idx * dim];
    for (int ik = 0; ik < num_ks; ++ik) {
        int k = ks[ik];
        int base = offsets[ik];
        float best_dist = FLT_MAX;
        int best_j = 0;
        for (int j = 0; j < k; ++j) {
            const float* cent = &centroids[(base + j) * dim];
            float dist = 0;
            for (int d = 0; d < dim; ++d) {
                float diff = pt[d] - cent[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }
        labels[idx * num_ks + ik] = best_j;
    }
}

__global__ void update_centroids_multiK(
    const float* data, int n, int dim,
    const int* labels,
    float* new_centroids, int* counts,
    const int* ks, const int* offsets,
    int num_ks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float* pt = &data[idx * dim];
    for (int ik = 0; ik < num_ks; ++ik) {
        int k = ks[ik];
        int base = offsets[ik];
        int lab = labels[idx * num_ks + ik];
        int cidx = base + lab;
        for (int d = 0; d < dim; ++d) {
            atomicAdd(&new_centroids[cidx * dim + d], pt[d]);
        }
        atomicAdd(&counts[cidx], 1);
    }
}


int main() {
    std::string base_dir = "output_results_omp";
    std::vector<std::string> classes = {"benign", "malignant", "normal"};

    auto files = collect_feature_files(base_dir, classes);
    if (files.empty()) {
        std::cerr << "Nenhum arquivo de features encontrado em " << base_dir << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> h_data;
    int total_n = 0, dim = 0;
    for (const auto& fpath : files) {
        int n_i, dim_i;
        auto v = load_features(fpath, n_i, dim_i);
        if (v.empty()) continue; // Pula arquivos vazios
        if (dim == 0 && !v.empty()) dim = dim_i;
        else if (dim_i != dim && !v.empty()) {
            std::cerr << "Dimensao inconsistente em " << fpath << std::endl;
            return EXIT_FAILURE;
        }
        h_data.insert(h_data.end(), v.begin(), v.end());
        total_n += n_i;
    }
    int n = total_n / dim; // n é o número de pontos
    std::cout << "Total de pontos carregados: " << n << " (dim=" << dim << ")\n";
    if (n == 0) {
        std::cerr << "Nenhum ponto de dado para processar." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int> ks = {2, 3, 4, 5};
    int num_ks = ks.size();
    std::vector<int> offsets(num_ks + 1, 0);
    for (int i = 0; i < num_ks; ++i) offsets[i+1] = offsets[i] + ks[i];
    int K_total = offsets.back();
    std::vector<float> h_centroids(K_total * dim);
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist_pt(0, n - 1);
    for (int ik = 0; ik < num_ks; ++ik) {
        int base = offsets[ik];
        for (int j = 0; j < ks[ik]; ++j) {
            int idx = dist_pt(rng);
            for (int d=0; d<dim; ++d)
                h_centroids[(base + j) * dim + d] = h_data[idx * dim + d];
        }
    }
    float *d_data, *d_centroids, *d_new_centroids;
    int *d_ks, *d_offsets, *d_labels, *d_counts;
    cudaMalloc(&d_data, n * dim * sizeof(float));
    cudaMalloc(&d_centroids, K_total * dim * sizeof(float));
    cudaMalloc(&d_new_centroids, K_total * dim * sizeof(float));
    cudaMalloc(&d_ks, num_ks * sizeof(int));
    cudaMalloc(&d_offsets, (num_ks + 1) * sizeof(int));
    cudaMalloc(&d_labels, n * num_ks * sizeof(int));
    cudaMalloc(&d_counts, K_total * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), n * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), K_total * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks, ks.data(), num_ks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), (num_ks + 1) * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    std::vector<float> h_new_centroids(K_total * dim);
    std::vector<int>   h_counts(K_total);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iter = 0;
    bool converged = false;

    //grava o evento de início antes do loop
    cudaEventRecord(start);

    while (iter < MAX_ITER && !converged) {
        cudaMemset(d_new_centroids, 0, K_total * dim * sizeof(float));
        cudaMemset(d_counts, 0, K_total * sizeof(int));

        assign_clusters_multiK<<<grid, block>>>(d_data, n, dim, d_centroids, d_ks, d_offsets, num_ks, d_labels);
        update_centroids_multiK<<<grid, block>>>(d_data, n, dim, d_labels, d_new_centroids, d_counts, d_ks, d_offsets, num_ks);

        cudaMemcpy(h_new_centroids.data(), d_new_centroids, K_total * dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts.data(), d_counts, K_total * sizeof(int), cudaMemcpyDeviceToHost);

        converged = true;
        for (int ik = 0; ik < num_ks; ++ik) {
            int base = offsets[ik];
            for (int j = 0; j < ks[ik]; ++j) {
                int idx = base + j;
                if (h_counts[idx] > 0) {
                    for(int d=0; d<dim; ++d){
                        float newc = h_new_centroids[idx * dim + d] / h_counts[idx];
                        if (std::fabs(newc - h_centroids[idx * dim + d]) > EPS) converged = false;
                        h_centroids[idx * dim + d] = newc;
                    }
                }
            }
        }
        cudaMemcpy(d_centroids, h_centroids.data(), K_total * dim * sizeof(float), cudaMemcpyHostToDevice);
        ++iter;
    }
    
    //grava o evento de fim e sincronizar
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //calcular e exibir o tempo
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Convergiu em " << iter << " iteracoes.\n";
    
    std::cout << "\n--- Desempenho da Execução ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tempo total do K-Means (GPU): " << milliseconds << " ms (" << milliseconds / 1000.0 << " s)\n";
    std::cout << "------------------------------\n\n";

    std::cout << "Centroides Finais:\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int ik = 0; ik < num_ks; ++ik) {
        std::cout << "K=" << ks[ik] << ": ";
        int base = offsets[ik];
        for (int j = 0; j < ks[ik]; ++j) {
            for(int d=0; d<dim; ++d)
                std::cout << h_centroids[(base + j) * dim + d] << " ";
        }
        std::cout << "\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_ks);
    cudaFree(d_offsets);
    cudaFree(d_labels);
    cudaFree(d_counts);

    return 0;
}