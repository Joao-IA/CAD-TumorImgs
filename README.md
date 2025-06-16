# CAD-TumorImgs
Trabalho Final de Computação de Alto Desempenho com objetivo de obter uma melhoria em um código existente proposto pelo artigo: (BLANK)


Para compilar o arquivo em CUDA e conseguir rodar ele, faça:
```
nvcc -std=c++17 -o process_dataset process_dataset.cu $(pkg-config --cflags --libs opencv4)
```

e então:
```
./process_dataset
```

Já para compilar o arquivo em CPP do OpenMP, faça:
```
g++ -std=c++17 -O3 -o process_dataset_omp process_dataset_omp.cpp $(pkg-config --cflags --libs opencv4) -fopenmp
```

e então:
```
./process_dataset_omp
```