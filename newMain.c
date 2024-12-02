#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#define INCREMENTO 500  // Incremento do tamanho do vetor a cada alocação extra

int h = 2;  // Altura do vetor
int w = 100; // Largura do vetor
int k = 10;  // Número de vizinhos mais próximos a considerar

// Função para ler um arquivo e armazenar os dados em um vetor
int ler_arquivo(const char *nome_arquivo, float **vetor) {
    FILE *arquivo = fopen(nome_arquivo, "r");  // Abre o arquivo para leitura
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo %s\n", nome_arquivo);
        return -1;  // Retorna -1 em caso de erro na abertura
    }

    int i = 0;
    int capacidade = INCREMENTO;  // Capacidade inicial do vetor
    *vetor = (float *)malloc(capacidade * sizeof(float));  // Alocando memória inicialmente
    if (*vetor == NULL) {
        printf("Erro de alocação de memória\n");
        fclose(arquivo);
        return -1;  // Retorna -1 se a alocação falhar
    }

    // Lendo os valores do arquivo e redimensionando a memória conforme necessário
    while (fscanf(arquivo, "%f", &(*vetor)[i]) != EOF) {
        i++;

        // Verificar se a capacidade foi atingida e alocar mais memória
        if (i >= capacidade) {
            capacidade += INCREMENTO;  // Aumentando a capacidade
            *vetor = (float *)realloc(*vetor, capacidade * sizeof(float));  // Realocando
            if (*vetor == NULL) {
                printf("Erro ao realocar memória\n");
                fclose(arquivo);
                return -1;  // Retorna -1 se a realocação falhar
            }
        }
    }

    fclose(arquivo);  // Fecha o arquivo após a leitura
    return i;  // Retorna a quantidade de valores lidos
}

// Função para calcular os vizinhos mais próximos (KNN)
float* knn(float *xtrain, float *xtest, float *ytrain, int trainSize, int testSize, int linhasTrain, int linhasTest) {
    float* ytest = malloc(linhasTest * sizeof(float));  // Alocando memória para os resultados

    float distancias[linhasTest][k];  // Matriz para armazenar as distâncias
    int indices[linhasTest][k];  // Matriz para armazenar os índices dos vizinhos

    // Inicializando as distâncias e índices
    for (int i = 0; i < linhasTest; i++) {
        for (int j = 0; j < k; j++) {
            distancias[i][j] = FLT_MAX;  // Inicializa com o valor máximo
            indices[i][j] = -1;  // Inicializa com -1
        }
    }

    float dist = 0;  // Variável para armazenar a distância calculada
    float media = 0;  // Variável para calcular a média dos vizinhos
    int i, j, l, m, n; //variáveis de loops

    // Início da medição de tempo
    clock_t start = clock();  // Captura o tempo inicial desconsidera inicialização e declarações de variáveis

    // Calculando as distâncias entre os pontos de teste e os pontos de treinamento
    #pragma omp parallel for private(dist, l, m, n)  // Paraleliza o loop externo
    for ( i = 0; i < linhasTest; i++) {
        for ( j = 0; j < linhasTrain; j++) {
            dist = 0;  // Reseta a distância para cada novo ponto de teste
            for ( l = 0; l < w; l++) {
                dist += pow(xtest[i * h + l] - xtrain[j * h + l], 2);  // Calcula a distância euclidiana
            }
            dist = sqrt(dist);  // Calcula a raiz quadrada da soma das diferenças ao quadrado
            for ( m = 0; m < k; m++) {
                // Se a nova distância for menor que a maior distância atual
                if (dist < distancias[i][m]) {
                    // Move os vizinhos para a direita
                    for ( n = k - 1; n > m; n--) {
                        distancias[i][n] = distancias[i][n - 1];
                        indices[i][n] = indices[i][n - 1];
                    }
                    distancias[i][m] = dist;  // Atualiza a nova distância
                    indices[i][m] = j;  // Atualiza o índice do vizinho
                    break;  // Sai do loop
                }
            }
        }
    }

    // Calculando a média dos valores dos vizinhos mais próximos
    #pragma omp parallel for reduction(+:media) private(j)  // Paraleliza o loop externo e usa redução
    for ( i = 0; i < linhasTest; i++) {
        media = 0;  // Reseta a média para cada novo ponto de teste
        for ( j = 0; j < k; j++) {
            media += ytrain[indices[i][j]];  // Soma os valores dos vizinhos
        }
        media = media / k;  // Calcula a média
        ytest[i] = media;  // Armazena o resultado
    }
    // Fim da medição de tempo
    clock_t end = clock();  // Captura o tempo final
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calcula o tempo de CPU em segundos
    printf("Tempo de execucao: %f segundos\n", cpu_time_used);  // Exibe o tempo de execução
    
    return ytest;  // Retorna os resultados
}

int main(int argc, char *argv[]) {
    // Verifica se o número correto de argumentos foi passado
    if (argc != 3) {
        printf("Uso incorreto. Exemplo de uso: %s <arquivoTrain.txt> <arquivoTest.txt>\n", argv[0]);
        return 1;  // Retorna erro se não houver dois arquivos fornecidos
    }
    
    // Definindo o número de threads
    omp_set_num_threads(2);

    float *xtrain = NULL, *xtest = NULL;  // Ponteiros para os vetores de treinamento e teste
    int tamTrain, tamTest;  // Variáveis para armazenar o tamanho dos vetores

    // Lendo os arquivos e armazenando os dados nos vetores
    tamTrain = ler_arquivo(argv[1], &xtrain);  // Usando o primeiro argumento para o arquivo de treinamento
    if (tamTrain == -1) return 1;  // Erro ao ler o arquivo de treinamento

    tamTest = ler_arquivo(argv[2], &xtest);  // Usando o segundo argumento para o arquivo de teste
    if (tamTest == -1) return 1;  // Erro ao ler o arquivo de teste

    // Cálculo do número de linhas a serem processadas
    int linhasTrain = ((tamTrain - w) / h) + 1;  // Calcula o número de linhas de treinamento
    int linhasTest = ((tamTest - w) / h) + 1;  // Calcula o número de linhas de teste

    float *ytrain = (float *)malloc(linhasTrain * sizeof(float));  // Alocação de memória para ytrain
    if (ytrain == NULL) {
        printf("Erro ao alocar memória para ytrain.\n");
        return 1;  // Retorna erro se a alocação falhar
    }

    // Copiando os últimos linhasTrain elementos de xtest para ytrain
    for (int i = 0; i < linhasTrain; i++) {
        ytrain[i] = xtest[tamTrain - linhasTrain + i];  // Cópia dos últimos linhasTrain elementos
    }

    // Chamada da função KNN
    float* resultados = knn(xtrain, xtest, ytrain, tamTrain, tamTest, linhasTrain, linhasTest);
    // para o relatório: com os arquivos disponibilizados e as variáveis globais h = 2, w = 100 e k = 10 temos 190 linhas em cada matriz (train e test)
    //                   assim, estamos trabalhando com matrizes de dimenções 100 x 190 e as de ídices e distâncias com 190 x 10.  
    // Liberando a memória alocada
    free(xtrain);
    free(xtest);
    free(ytrain);  
    free(resultados);  

    return 0;  // Retorna 0 para indicar que o programa terminou com sucesso
}
