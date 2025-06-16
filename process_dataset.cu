#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <algorithm>
#include <filesystem> // Para navegar e criar diretórios (requer C++17)

namespace fs = std::filesystem;

// Constante para o tamanho do warp a ser usada no código do host (CPU)
const int HOST_DEFINED_WARP_SIZE = 32;

// O kernel CUDA permanece o mesmo
__global__ void haralick_texture_kernel(cudaTextureObject_t tex, float *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float center = tex2D<float>(tex, x, y);
    float right = (x + 1 < width) ? tex2D<float>(tex, x + 1, y) : center;

    float val = (center + right) * 0.5f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) { // warpSize aqui é o do device
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if ((threadIdx.y * blockDim.x + threadIdx.x) % warpSize == 0) { // warpSize aqui é o do device
        int idx = (y * width + x) / warpSize; // warpSize aqui é o do device
        out[idx] = val;
    }
}

// FUNÇÃO: Para salvar a imagem de saída
void saveOutputImage(const std::vector<float>& h_out, int width, int height, const std::string& outputPath) {
    int newW_cuda = width / HOST_DEFINED_WARP_SIZE;
    if (newW_cuda <= 0) return;

    cv::Mat outImg_cuda(height, newW_cuda, CV_32F);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < newW_cuda; ++c) {
            int idx = r * newW_cuda + c;
            if (idx < h_out.size()) {
                outImg_cuda.at<float>(r, c) = h_out[idx];
            }
        }
    }

    cv::Mat outImg8U_cuda;
    cv::normalize(outImg_cuda, outImg_cuda, 0, 255, cv::NORM_MINMAX);
    outImg_cuda.convertTo(outImg8U_cuda, CV_8U);

    if (cv::imwrite(outputPath, outImg8U_cuda)) {
        std::cout << "  -> Imagem salva em: " << outputPath << std::endl;
    } else {
        std::cerr << "  -> FALHA ao salvar imagem em: " << outputPath << std::endl;
    }
}

// FUNÇÃO: Encontra todos os arquivos de imagem nos subdiretórios
void findImageFiles(const std::string& rootPath, std::vector<std::string>& imagePaths) {
    std::vector<std::string> subdirs = {"benign", "malignant", "normal"};
    for (const auto& subdir : subdirs) {
        fs::path dirPath = fs::path(rootPath) / subdir;
        if (!fs::exists(dirPath)) {
            std::cout << "Aviso: Diretório não encontrado: " << dirPath << std::endl;
            continue;
        }
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // Verifica extensões comuns de imagem
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
                    imagePaths.push_back(entry.path().string());
                }
            }
        }
    }
}


int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(8);

    // --- CONFIGURAÇÃO ---
    std::string root_dataset_path = "./dataset_busi_drive/Dataset_BUSI_with_GT"; // Caminho base para os diretórios
    if (argc > 1) {
        root_dataset_path = argv[1];
    }
    const std::string output_directory = "./output_results"; // Diretório para salvar os resultados

    // *** PONTO CHAVE 1: Cria o diretório de saída principal se ele não existir. ***
    // A função create_directories não gera erro se o diretório já existir.
    fs::create_directories(output_directory);

    // --- COLETA DE ARQUIVOS ---
    std::vector<std::string> imagePaths;
    std::cout << "Procurando imagens em: " << root_dataset_path << std::endl;
    findImageFiles(root_dataset_path, imagePaths);

    if (imagePaths.empty()) {
        std::cerr << "Nenhuma imagem encontrada. Verifique o caminho do dataset." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Total de imagens encontradas: " << imagePaths.size() << std::endl;

    // --- PROCESSAMENTO EM LOTE COM CUDA STREAMS ---
    const int num_streams = 4; // Número de streams paralelas. 2 a 4 é um bom começo.
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total);

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        int stream_idx = i % num_streams;
        cudaStream_t current_stream = streams[stream_idx];
        
        std::cout << "\nProcessando imagem " << (i + 1) << "/" << imagePaths.size() << ": " << imagePaths[i] << " (na stream " << stream_idx << ")" << std::endl;

        // 1. Carregar a imagem no Host (CPU)
        cv::Mat img_orig = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        if (img_orig.empty()) {
            std::cerr << "  -> Erro: Falha ao carregar a imagem." << std::endl;
            continue;
        }
        int width = img_orig.cols;
        int height = img_orig.rows;
        
        cv::Mat imgF_for_processing;
        img_orig.convertTo(imgF_for_processing, CV_32F, 1.0 / 255.0);

        // 2. Alocar memória na GPU (Device)
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, width, height);
        
        size_t srcPitch = width * sizeof(float);
        cudaMemcpy2DToArrayAsync(cuArray, 0, 0, imgF_for_processing.ptr<float>(), srcPitch, width * sizeof(float), height, cudaMemcpyHostToDevice, current_stream);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp; texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint; texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

        int outSize = (width * height) / HOST_DEFINED_WARP_SIZE;
        if (outSize <= 0) {
             std::cerr << "  -> Erro: Imagem muito pequena para processar." << std::endl;
             cudaDestroyTextureObject(texObj);
             cudaFreeArray(cuArray);
             continue;
        }
        float *d_out;
        cudaMalloc(&d_out, outSize * sizeof(float));

        // 3. Lançar o Kernel na Stream
        dim3 block(32, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        haralick_texture_kernel<<<grid, block, 0, current_stream>>>(texObj, d_out, width, height);

        // 4. Copiar resultados de volta para o Host (CPU)
        std::vector<float> h_out(outSize);
        cudaMemcpyAsync(h_out.data(), d_out, outSize * sizeof(float), cudaMemcpyDeviceToHost, current_stream);

        // 5. Sincronizar *esta* stream específica para garantir que o processamento terminou antes de salvar
        cudaStreamSynchronize(current_stream);

        // 6. Construir o caminho e salvar o resultado
        fs::path inputPath(imagePaths[i]);
        std::string outputFilename = inputPath.stem().string() + "_haralick_out.png";
        fs::path outputPath = fs::path(output_directory) / inputPath.parent_path().filename() / outputFilename;
        
        // *** PONTO CHAVE 2: Cria o subdiretório de categoria (ex: ./output_results/benign) se não existir. ***
        fs::create_directories(outputPath.parent_path());
        
        saveOutputImage(h_out, width, height, outputPath.string());

        // 7. Limpar recursos da GPU para esta imagem
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        cudaFree(d_out);
    }

    // Sincronização final e medição de tempo
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    float total_milliseconds = 0;
    cudaEventElapsedTime(&total_milliseconds, start_total, stop_total);
    
    std::cout << "\n--- Processamento concluído ---" << std::endl;
    std::cout << "Tempo total de execução para " << imagePaths.size() << " imagens: " << total_milliseconds / 1000.0 << " segundos." << std::endl;
    std::cout << "Média por imagem: " << total_milliseconds / imagePaths.size() << " ms." << std::endl;

    // Cleanup final
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);

    return EXIT_SUCCESS;
}