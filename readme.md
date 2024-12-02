# Projeto KNN em C

Este projeto implementa um algoritmo de K-Nearest Neighbors (KNN) em C, que lê dados de arquivos de texto e realiza a classificação com base nos vizinhos mais próximos.

## Descrição

O programa lê dois arquivos de texto:
- Um arquivo de treinamento (`arquivoTrain.txt`)
- Um arquivo de teste (`arquivoTest.txt`)

Os dados são armazenados em vetores e o algoritmo KNN é utilizado para prever os valores do conjunto de teste com base nos dados do conjunto de treinamento.

## Estrutura do Código

O código é dividido em várias funções principais:

1. **ler_arquivo**: Lê os dados de um arquivo e armazena em um vetor dinâmico. A função também gerencia a alocação de memória, aumentando a capacidade do vetor conforme necessário.

2. **knn**: Implementa o algoritmo KNN. Calcula as distâncias entre os pontos de teste e os pontos de treinamento, seleciona os k vizinhos mais próximos e calcula a média dos valores correspondentes para prever os resultados.

3. **main**: Função principal que gerencia a execução do programa. Lê os arquivos de entrada, chama a função KNN e libera a memória alocada.

## Compilação e Execução

Para compilar o programa, utilize um compilador C, como `gcc`. Execute o seguinte comando no terminal:
    gcc newMain.c -o knn -lm -fopenmp
    ./knn arquivoTrain.txt arquivoTest.txt

## Exemplo de Uso

1. Crie um arquivo `arquivoTrain.txt` com os dados de treinamento.
2. Crie um arquivo `arquivoTest.txt` com os dados de teste.
3. Execute o programa conforme descrito acima.

### Formato dos Arquivos

- **arquivoTrain.txt**: Contém os dados de treinamento, onde cada linha representa um valor numérico.
- **arquivoTest.txt**: Contém os dados de teste, seguindo o mesmo formato do arquivo de treinamento.

## Dependências

- `math.h`: Para funções matemáticas como `pow` e `sqrt`.
- `float.h`: Para carregar FLT_MAX como um valor semlhante a infinito. 
- `stdio.h`: Para operações de entrada e saída.
- `stdlib.h`: Para alocação de memória dinâmica.
- `omp.h`: Para paralelização do código usando OpenMP.

## Licença

Este projeto é de domínio público. Sinta-se à vontade para usar e modificar conforme necessário.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir um problema ou enviar um pull request.
