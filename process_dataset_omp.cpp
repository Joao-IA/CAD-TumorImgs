#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <omp.h>       // Para OpenMP
#include <opencv2/opencv.hpp>
#include <filesystem>  // NOVO: Para navegar e criar diretórios

namespace fs = std::filesystem;

// Constante para o tamanho da agregação
const int AGGREGATION_SIZE = 32;

// Função para salvar um std::vector<float> em um arquivo de texto
void saveVectorToFile(const std::vector<float>& data, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << std::fixed << std::setprecision(8);
        for (const auto& val : data) {
            outFile << val << std::endl;
        }
        outFile.close();
        std::cout << "  -> Vetor de saída bruto salvo em: " << filename << std::endl;
    } else {
        std::cerr << "  -> ERRO: Não foi possível abrir o arquivo para salvar: " << filename << std::endl;
    }
}

// Função de processamento OpenMP (sem alterações internas)
void haralick_texture_openmp(const cv::Mat& imgF, std::vector<float>& h_out_omp, int aggregation_size) {
    int height = imgF.rows;
    int width = imgF.cols;

    if (height == 0 || width == 0) { h_out_omp.clear(); return; }

    int newW = width / aggregation_size;
    if (newW == 0 && width > 0) newW = 1;

    h_out_omp.assign(height * newW, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < height; ++r) {
        const float* row_ptr = imgF.ptr<float>(r);
        for (int c_out = 0; c_out < newW; ++c_out) {
            float current_segment_sum = 0.0f;
            int start_x_original = c_out * aggregation_size;

            for (int i = 0; i < aggregation_size; ++i) {
                int x_original = start_x_original + i;
                if (x_original >= width) break;

                float center_val = row_ptr[x_original];
                float right_val = (x_original + 1 < width) ? row_ptr[x_original + 1] : center_val;

                float term = (center_val + right_val) * 0.5f;
                current_segment_sum += term;
            }
            if (r * newW + c_out < h_out_omp.size()) {
                 h_out_omp[r * newW + c_out] = current_segment_sum;
            }
        }
    }
}

// NOVA FUNÇÃO: Para salvar a imagem de saída de forma organizada
void saveOutputImage_omp(const std::vector<float>& h_out, int width, int height, const std::string& outputPath) {
    int newW_omp = width / AGGREGATION_SIZE;
    if (newW_omp <= 0) return;

    // --- CORREÇÃO AQUI ---
    // 1. Crie uma matriz vazia com o tamanho correto. O OpenCV alocará a memória para ela.
    cv::Mat outImg_omp(height, newW_omp, CV_32F);

    // 2. Preencha a matriz copiando os dados do vetor.
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < newW_omp; ++c) {
            int idx = r * newW_omp + c;
            if (idx < h_out.size()) {
                outImg_omp.at<float>(r, c) = h_out[idx];
            } else {
                // Fallback de segurança, não deve acontecer em condições normais
                outImg_omp.at<float>(r, c) = 0.0f;
            }
        }
    }
    // --- FIM DA CORREÇÃO ---

    cv::Mat outImg8U_omp;
    cv::normalize(outImg_omp, outImg_omp, 0, 255, cv::NORM_MINMAX);
    outImg_omp.convertTo(outImg8U_omp, CV_8U);

    if (cv::imwrite(outputPath, outImg8U_omp)) {
        std::cout << "  -> Imagem salva em: " << outputPath << std::endl;
    } else {
        std::cerr << "  -> FALHA ao salvar imagem em: " << outputPath << std::endl;
    }
}

// NOVA FUNÇÃO: Encontra todos os arquivos de imagem nos subdiretórios
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
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif") {
                    imagePaths.push_back(entry.path().string());
                }
            }
        }
    }
}

// --- MAIN REFATORADO PARA PROCESSAMENTO EM LOTE ---
int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(3);

    // --- CONFIGURAÇÃO ---
    std::string root_dataset_path = "./dataset_busi_drive/Dataset_BUSI_with_GT";
    if (argc > 1) {
        root_dataset_path = argv[1];
    }
    const std::string output_directory = "./output_results_omp";

    // Cria o diretório de saída principal se ele não existir
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

    // --- PROCESSAMENTO EM LOTE COM OPENMP ---
    auto start_total = std::chrono::high_resolution_clock::now();

    for (const auto& image_path : imagePaths) {
        std::cout << "\nProcessando imagem: " << image_path << std::endl;

        cv::Mat img_orig = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img_orig.empty()) {
            std::cerr << "  -> Erro: Falha ao carregar a imagem." << std::endl;
            continue;
        }

        cv::Mat imgF;
        img_orig.convertTo(imgF, CV_32F, 1.0 / 255.0);

        std::vector<float> h_out_omp;
        haralick_texture_openmp(imgF, h_out_omp, AGGREGATION_SIZE);

        if (h_out_omp.empty()) {
            std::cerr << "  -> Erro: Processamento OpenMP resultou em saída vazia." << std::endl;
            continue;
        }

        // --- Construção do caminho e salvamento dos resultados ---
        fs::path inputPath(image_path);
        std::string outputFilename_img = inputPath.stem().string() + "_haralick_out_omp.png";
        std::string outputFilename_txt = inputPath.stem().string() + "_haralick_raw_omp.txt";

        fs::path outputPath_img = fs::path(output_directory) / inputPath.parent_path().filename() / outputFilename_img;
        fs::path outputPath_txt = fs::path(output_directory) / inputPath.parent_path().filename() / outputFilename_txt;

        // Garante que o subdiretório de destino exista
        fs::create_directories(outputPath_img.parent_path());

        saveOutputImage_omp(h_out_omp, img_orig.cols, img_orig.rows, outputPath_img.string());
        saveVectorToFile(h_out_omp, outputPath_txt.string());
    }

    auto stop_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(stop_total - start_total);

    std::cout << "\n--- Processamento concluído ---" << std::endl;
    std::cout << "Tempo total de execução para " << imagePaths.size() << " imagens: "
              << duration_total.count() / 1000.0 << " segundos." << std::endl;
    std::cout << "Média por imagem: " << static_cast<double>(duration_total.count()) / imagePaths.size()
              << " ms." << std::endl;

    return EXIT_SUCCESS;
}