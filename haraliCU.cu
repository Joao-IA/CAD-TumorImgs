// haralick_optimized.cu
// Compilação: nvcc -o haralick_optimized haralick_optimized.cu `pkg-config --cflags --libs opencv4`

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cooperative_groups;

// Texture object para leitura eficiente
cudaTextureObject_t texObj = 0;

// Kernel otimizado usando memória de textura e warp-shuffle
__global__ void haralick_texture_kernel(float *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Leitura via textura (L1 cache)
    float center = tex2D<float>(texObj, x, y);
    float right  = (x + 1 < width) ? tex2D<float>(texObj, x + 1, y) : center;

    // Exemplo de comunicação intra-warp: soma com shuffle
    float val = (center + right) * 0.5f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Cada warp write em saídas distintas
    int warpIdx = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    if ((threadIdx.y * blockDim.x + threadIdx.x) % warpSize == 0) {
        int idx = (y * width + x) / warpSize;
        out[idx] = val;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <imagem_entrada>" << std::endl;
        return EXIT_FAILURE;
    }

    // Carregar imagem em tons de cinza
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Erro ao carregar imagem." << std::endl;
        return EXIT_FAILURE;
    }

    int width  = img.cols;
    int height = img.rows;

    // Converter para float normalizado [0,1]
    cv::Mat imgF;
    img.convertTo(imgF, CV_32F, 1.0 / 255.0);

    // Alocar array CUDA e copiar dados
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, imgF.ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Configurar recurso de textura
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]    = cudaAddressModeClamp;
    texDesc.addressMode[1]    = cudaAddressModeClamp;
    texDesc.filterMode        = cudaFilterModePoint;
    texDesc.readMode          = cudaReadModeElementType;
    texDesc.normalizedCoords  = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Alocar saída (tamanho reduzido por warp)
    int outSize = (width * height) / warpSize;
    float *d_out;
    cudaMalloc(&d_out, outSize * sizeof(float));

    // Configurar grid e bloco
    dim3 block(32, 8); // 256 threads por bloco
    dim3 grid( (width + block.x - 1) / block.x,
               (height + block.y - 1) / block.y);

    // Launch
    haralick_texture_kernel<<<grid, block>>>(d_out, width, height);
    cudaDeviceSynchronize();

    // Transferir resultado de volta
    std::vector<float> h_out(outSize);
    cudaMemcpy(h_out.data(), d_out, outSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Exemplo simples de saída: salvar primeira dimensão como imagem
    int newW = width / warpSize;
    cv::Mat outImg(height, newW, CV_32F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < newW; ++x) {
            outImg.at<float>(y, x) = h_out[y*newW + x];
        }
    }
    outImg.convertTo(outImg, CV_8U, 255.0);
    cv::imwrite("haralick_out.png", outImg);

    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_out);

    std::cout << "Processamento concluído. Resultado salvo em haralick_out.png" << std::endl;
    return EXIT_SUCCESS;
}
