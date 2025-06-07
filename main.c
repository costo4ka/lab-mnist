#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>

#define MAX_LAYERS 10
#define INPUT_SIZE 784
#define OUTPUT_CLASSES 10

// пути к бинарникам MNIST
#define MNIST_IMAGE_FILE "/home/kabanpunk/CLionProjects/NeuralNetwork/MNIST/raw/train-images-idx3-ubyte"
#define MNIST_LABEL_FILE "/home/kabanpunk/CLionProjects/NeuralNetwork/MNIST/raw/train-labels-idx1-ubyte"

// параметры для синтетических данных (по умолчанию используется MNIST, но можно переключиться на синтетические данные, если нужно.)
#define SYNTHETIC_SAMPLES 1000
#define SYNTHETIC_TEST_SAMPLES 200

// Размер мини-батча (веса сети будут обновляться после обработки каждых 32 образцов данных)
#define BATCH_SIZE 32

// роль EARLY_STOP_PATIENCE: если потери не улучшаются в течение определенного количества эпох (5), обучение прерывается
#define EARLY_STOP_PATIENCE 5

// безопасное прерывание обучения
volatile sig_atomic_t stopTrainingFlag = 0;

void handle_sigint(int sig) {
    stopTrainingFlag = 1;
}

// -------------------------- Структуры -------------------------------
// описание одного слоя сети
typedef struct {
    int numNeurons;       // число нейронов в слое
    double *neurons;      // значения активаций
    double *biases;       // bias
    double **weights;     // веса: [numNeurons][prevLayerNeurons]
    double *z;            // z = w·x + b
    double *delta;        // локальные градиенты
} Layer;

// сеть целиком
typedef struct {
    int numLayers;        // общее число слоёв (включая входной, скрытые и выходной)
    Layer *layers;        // массив слоёв
    int *layerSizes;      // размеры слоёв, прочитанные из config.txt
    double learningRate;  // скорость обучения
    double regularization;// коэффициент L2-регуляризации
} Network;

// буквально тест-кейс
typedef struct {
    double image[INPUT_SIZE];
    double target[OUTPUT_CLASSES]; // one-hot вектор
} TestCase;

// метрики
typedef struct {
    double loss; // далее указана в computeLoss (во время  обучения)
    double accuracy; // число правильных предсказаний/общее число примеров (общая простая оценка)
    double precision; // точность для положительного класса (когда важны ложноположитеьные)
    double recall; // полнота (ложноотрицательные)
} Metrics;

/**
 * @brief  Случайное число из распределения N(mean, stddev)
 *
 * @param  mean    
математическое ожидание
 * @param  stddev  среднеквадратичное отклонение
 * @return double  сгенерированное значение
 */
double randn(double mean, double stddev) {
    double u1 = ((double) rand() + 1) / ((double) RAND_MAX + 1);
    double u2 = ((double) rand() + 1) / ((double) RAND_MAX + 1);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
    return z0 * stddev + mean;
}

/**
 * @brief  Чтение файла config.txt и заполнение параметров сети
 *
 * Формат файла:
 *   neurons: 784,128,64,10
 *   learning_rate: 0.001
 *   regularization: 0.0005
 *
 * @param  filename        путь к config.txt
 * @param  layerSizes[out] массив с размерами слоёв (выделяется внутри)
 * @param  numLayers[out]  количество слоёв
 * @param  learningRate[out] скорость обучения
 * @param  regularization[out] коэффициент L2-регуляризации
 */
void readConfig(const char *filename, int **layerSizes, int *numLayers, double *learningRate, double *regularization) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open config file");
        exit(EXIT_FAILURE);
    }
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "neurons:", 8) == 0) {
            char *ptr = line + 8;
            int count = 1;
            for (char *c = ptr; *c; c++) {
                if (*c == ',') count++;
            }
            *numLayers = count;
            *layerSizes = (int*)malloc(sizeof(int) * count);
            char *token = strtok(ptr, ", \n");
            int idx = 0;
            while (token) {
                (*layerSizes)[idx++] = atoi(token);
                token = strtok(NULL, ", \n");
            }
        } else if (strncmp(line, "learning_rate:", 14) == 0) {
            *learningRate = atof(line + 14);
        } else if (strncmp(line, "regularization:", 15) == 0) {
            *regularization = atof(line + 15);
        }
    }
    fclose(fp);
}

// -------------------------- Функции активации -------------------------------

double relu(double x) { // ля всех скрытых
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

void softmax(double *z, double *output, int n) { // для выходного слоя
    double max = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > max)
            max = z[i];
    }
    double sum = 0;
    for (int i = 0; i < n; i++) {
        output[i] = exp(z[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

/**
 * @brief  Создаёт один слой нейросети со случайной инициализацией весов
 *
 * @param  numNeurons      количество нейронов в слое
 * @param  prevNeurons     размер предыдущего слоя (0 для входного)
 * @return Layer           полностью готовая структура Layer
 */
Layer createLayer(int numNeurons, int prevNeurons) {
    Layer layer;
    layer.numNeurons = numNeurons;
    layer.neurons = (double*)calloc(numNeurons, sizeof(double));
    layer.biases = (double*)calloc(numNeurons, sizeof(double));
    layer.z = (double*)calloc(numNeurons, sizeof(double));
    layer.delta = (double*)calloc(numNeurons, sizeof(double));
    if (prevNeurons > 0) {
        layer.weights = (double**)malloc(numNeurons * sizeof(double*));
        for (int i = 0; i < numNeurons; i++) {
            layer.weights[i] = (double*)malloc(prevNeurons * sizeof(double));
            for (int j = 0; j < prevNeurons; j++) {
                double stddev = sqrt(2.0 / prevNeurons);
                layer.weights[i][j] = randn(0, stddev);
            }
        }
    } else {
        layer.weights = NULL;
    }
    return layer;
}

/**
 * @brief  Конструирует сеть из описания слоёв
 *
 * @param  layerSizes      массив размеров слоёв
 * @param  numLayers       количество слоёв
 * @param  learningRate    скорость обучения
 * @param  regularization  коэффициент L2-регуляризации
 * @return Network         полностью инициализированная сеть
 */
Network createNetwork(int *layerSizes, int numLayers, double learningRate, double regularization) {
    Network net;
    net.numLayers = numLayers;
    net.layerSizes = (int*)malloc(numLayers * sizeof(int));
    memcpy(net.layerSizes, layerSizes, numLayers * sizeof(int));
    net.learningRate = learningRate;
    net.regularization = regularization;
    net.layers = (Layer*)malloc(numLayers * sizeof(Layer));
    for (int i = 0; i < numLayers; i++) {
        int prev = (i == 0) ? 0 : layerSizes[i-1];
        net.layers[i] = createLayer(layerSizes[i], prev);
    }
    return net;
}

/**
 * @brief  Освобождает всю связанную с сетью память
 *
 * @param  net  указатель на Network, созданный createNetwork()
 */
void freeNetwork(Network *net) {
    for (int i = 0; i < net->numLayers; i++) {
        free(net->layers[i].neurons);
        free(net->layers[i].biases);
        free(net->layers[i].z);
        free(net->layers[i].delta);
        if (net->layers[i].weights) {
            int numNeurons = net->layers[i].numNeurons;
            int prevNeurons = (i==0) ? 0 : net->layerSizes[i-1];
            for (int j = 0; j < numNeurons; j++) {
                free(net->layers[i].weights[j]);
            }
            free(net->layers[i].weights);
        }
    }
    free(net->layers);
    free(net->layerSizes);
}

/**
 * @brief  Сохраняет веса и гиперпараметры сети в текстовый файл
 *
 * @param  net       сеть
 * @param  filename  куда сохранять
 */
void saveWeights(Network *net, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file for saving weights");
        return;
    }
    fprintf(fp, "%d\n", net->numLayers);
    for (int i = 0; i < net->numLayers; i++) {
        fprintf(fp, "%d ", net->layerSizes[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "%.6f %.6f\n", net->learningRate, net->regularization);
    for (int l = 1; l < net->numLayers; l++) {
        Layer *curr = &net->layers[l];
        for (int i = 0; i < curr->numNeurons; i++) {
            fprintf(fp, "%.6f ", curr->biases[i]);
        }
        fprintf(fp, "\n");
        int prevNeurons = net->layerSizes[l - 1];
        for (int i = 0; i < curr->numNeurons; i++) {
            for (int j = 0; j < prevNeurons; j++) {
                fprintf(fp, "%.6f ", curr->weights[i][j]);
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
    printf("Weights successfully saved to %s\n", filename);
}

/**
 * @brief  Загружает веса и гиперпараметры из файла
 *
 * @param  net       уже созданная сеть (размеры должны совпадать)
 * @param  filename  источник данных
 */
void loadWeights(Network *net, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Не удалось открыть файл для загрузки весов");
        return;
    }
    int numLayers;
    fscanf(fp, "%d", &numLayers);
    if (numLayers != net->numLayers) {
        fprintf(stderr, "Несовпадение числа слоёв: файл=%d, сеть=%d\n", numLayers, net->numLayers);
        fclose(fp);
        return;
    }
    for (int i = 0; i < net->numLayers; i++) {
        int size;
        fscanf(fp, "%d", &size);
        if (size != net->layerSizes[i]) {
            fprintf(stderr, "Несовпадение размера слоя %d: файл=%d, сеть=%d\n", i, size, net->layerSizes[i]);
            fclose(fp);
            return;
        }
    }
    double learningRate, regularization;
    fscanf(fp, "%lf %lf", &learningRate, &regularization);
    net->learningRate = learningRate;
    net->regularization = regularization;
    for (int l = 1; l < net->numLayers; l++) {
        Layer *curr = &net->layers[l];
        for (int i = 0; i < curr->numNeurons; i++) {
            fscanf(fp, "%lf", &curr->biases[i]);
        }
        int prevNeurons = net->layerSizes[l - 1];
        for (int i = 0; i < curr->numNeurons; i++) {
            for (int j = 0; j < prevNeurons; j++) {
                fscanf(fp, "%lf", &curr->weights[i][j]);
            }
        }
    }
    fclose(fp);
    printf("Весы успешно загружены из %s\n", filename);
}

/**
 * @brief  Прямое распространение по всем слоям сети
 *
 * @param  net    сеть
 * @param  input  указатель на массив входных данных размера INPUT_SIZE
 */
void forwardPropagation(Network *net, double *input) {
    // копируем входные данные в «нейроны» первого слоя
    for (int i = 0; i < net->layerSizes[0]; i++) {
        net->layers[0].neurons[i] = input[i];
    }
    // проходим по всем остальным слоям подряд
    for (int l = 1; l < net->numLayers; l++) {
        Layer *prev = &net->layers[l-1];
        Layer *curr = &net->layers[l];
        // для каждого нейрона в текущем слое
        for (int i = 0; i < curr->numNeurons; i++) {
            // считаем взвешенную сумму входов + смещение (bias)
            double sum = curr->biases[i];
            for (int j = 0; j < prev->numNeurons; j++) {
                sum += curr->weights[i][j] * prev->neurons[j];
            }
            // сохраняем «сырое» значение
            curr->z[i] = sum;
            // если это не последний слой, применяем ReLU:
            // отрицательные значения превращаем в 0, положительные — оставляем
            if (l < net->numLayers - 1)
                curr->neurons[i] = relu(sum);
            else
                // на выходном слое пока просто сохраняем сумму,
                // softmax применится после цикла
                curr->neurons[i] = sum;
        }
        // после обработки полного последнего слоя —
        // превращаем все «сырые» суммы в вероятности по классам
        if (l == net->numLayers - 1) {
            softmax(curr->z, curr->neurons, curr->numNeurons);
        }
    }
}

/**
 * @brief  Вычисляет локальные градиенты δ для всех слоёв
 *
 * @param  net     сеть (после forwardPropagation)
 * @param  target  one-hot вектор истинной метки
 */
void computeDeltas(Network *net, double *target) {
    // вычисляем дельты на выходном слое:
    // delta = (предсказанная вероятность – истинная метка)
    Layer *outputLayer = &net->layers[net->numLayers - 1];
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        outputLayer->delta[i] = outputLayer->neurons[i] - target[i];
    }
    // проходим от предпоследнего к первому слою обратным циклом
    for (int l = net->numLayers - 2; l > 0; l--) {
        // в самом цикле считаем ошибку
        Layer *curr = &net->layers[l];
        Layer *next = &net->layers[l+1];
        for (int i = 0; i < curr->numNeurons; i++) {
            double error = 0;
            for (int k = 0; k < next->numNeurons; k++) {
                error += next->weights[k][i] * next->delta[k];
            }
            curr->delta[i] = error * relu_derivative(curr->z[i]);
        }
    }
}

/**
 * @brief  Градиентный шаг: обновляет веса и смещения сети
 *
 * @param  net        сеть
 * @param  gradW      накопленные градиенты весов   gradW[layer][i][j]
 * @param  gradB      накопленные градиенты bias'ов gradB[layer][i]
 * @param  batchSize  размер текущего мини-батча
 */
void updateWeights(Network *net, double ***gradW, double **gradB, int batchSize) {
    for (int l = 1; l < net->numLayers; l++) {
        Layer *curr = &net->layers[l];
        Layer *prev = &net->layers[l-1];
        for (int i = 0; i < curr->numNeurons; i++) {
            for (int j = 0; j < prev->numNeurons; j++) {
                curr->weights[i][j] -= net->learningRate * (gradW[l][i][j] / batchSize + net->regularization * curr->weights[i][j]);
            }
            curr->biases[i] -= net->learningRate * (gradB[l][i] / batchSize);
        }
    }
}

/**
 * @brief  Кросс-энтропия для одного примера
 *
 * @param  output  предсказанные вероятности softmax
 * @param  target  one-hot истинная метка
 * @param  n       длина векторов (OUTPUT_CLASSES)
 * @return double  значение loss
 */
double computeLoss(double *output, double *target, int n) {
    double loss = 0;
    for (int i = 0; i < n; i++) {
        loss -= target[i] * log(output[i] + 1e-8);
    }
    return loss;
}

/**
 * @brief  Добавляет шум в картинку
 *
 * @param  input  массив пикселей [0;1]
 * @param  n      количество пикселей
 */
void addNoise(double *input, int n) {
    // сколько пикселей поменять
    int numNoisy = n * (5 + rand() % 6) / 100;
    // накладываем шум
    for (int i = 0; i < numNoisy; i++) {
        int idx = rand() % n;
        input[idx] += (rand() % 2 == 0) ? 0.2 : -0.2;
        // добавляем границы
        if (input[idx] < 0) input[idx] = 0;
        if (input[idx] > 1) input[idx] = 1;
    }
}

/**
 * @brief  Генерирует синтетический объект и one-hot метку
 *
 * Создаётся бинарное изображение 28×28 с шумом;
 * случайный класс назначается целевой меткой.
 *
 * @param  input   массив для изображения (разм. INPUT_SIZE)
 * @param  target  массив метки (разм. OUTPUT_CLASSES)
 */
void generateSyntheticData(double *input, double *target) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = rand() % 2;
    }
    addNoise(input, INPUT_SIZE);
    int label = rand() % OUTPUT_CLASSES;
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        target[i] = (i == label) ? 1.0 : 0.0;
    }
}

// -------------------------- Чтение MNIST данных -------------------------------
// меняем порядок байт
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/**
 * @brief  Читает изображения из бинарного файла MNIST
 *
 * @param  filename          путь к файлу train-images-idx3-ubyte
 * @param  numberOfImages[out] количество картинок
 * @param  imageSize[out]      размер одной картинки (обычно 784)
 * @return double**            массив изображений [N][imageSize], нормированный в [0;1]
 */
double **loadMNISTImages(const char *filename, int *numberOfImages, int *imageSize) {
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        perror("Не удалось открыть файл с изображениями MNIST");
        exit(EXIT_FAILURE);
    }
    int magic = 0;
    fread(&magic, sizeof(int), 1, fp);
    magic = reverseInt(magic);
    if (magic != 2051) {
        fprintf(stderr, "Неверный формат файла изображений MNIST!\n");
        exit(EXIT_FAILURE);
    }
    // cчитываем число изображений
    fread(numberOfImages, sizeof(int), 1, fp);
    *numberOfImages = reverseInt(*numberOfImages);
    int rows = 0, cols = 0;
    // cчитываем число строк и столбцов (обычно 28 и 28)
    fread(&rows, sizeof(int), 1, fp);
    rows = reverseInt(rows);
    fread(&cols, sizeof(int), 1, fp);
    cols = reverseInt(cols);
    *imageSize = rows * cols;
    double **images = (double **)malloc((*numberOfImages) * sizeof(double *));
    for (int i = 0; i < *numberOfImages; i++) {
        images[i] = (double *)malloc((*imageSize) * sizeof(double));
        for (int j = 0; j < *imageSize; j++) {
            unsigned char temp = 0;
            fread(&temp, sizeof(unsigned char), 1, fp);
            images[i][j] = temp / 255.0;
        }
    }
    fclose(fp);
    return images;
}

/**
 * @brief  Читает метки из файла MNIST
 *
 * @param  filename            путь к train-labels-idx1-ubyte
 * @param  numberOfLabels[out] количество меток
 * @return unsigned char*      массив меток [N]
 */
unsigned char *loadMNISTLabels(const char *filename, int *numberOfLabels) {
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        perror("Не удалось открыть файл с метками MNIST");
        exit(EXIT_FAILURE);
    }
    int magic = 0;
    fread(&magic, sizeof(int), 1, fp);
    magic = reverseInt(magic);
    if (magic != 2049) {
        fprintf(stderr, "Неверный формат файла меток MNIST!\n");
        exit(EXIT_FAILURE);
    }
    fread(numberOfLabels, sizeof(int), 1, fp);
    *numberOfLabels = reverseInt(*numberOfLabels);
    unsigned char *labels = (unsigned char *)malloc((*numberOfLabels) * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), *numberOfLabels, fp);
    fclose(fp);
    return labels;
}

/**
 * @brief  Комплексная загрузка датасета MNIST
 *
 * @param  imgFile    путь к файлу изображений
 * @param  labelFile  путь к файлу меток
 * @param  dataSize[out] число примеров
 * @return TestCase*  массив структур TestCase [dataSize]
 */
TestCase* loadMNISTData(const char *imgFile, const char *labelFile, int *dataSize) {
    int numImages, imageSize;
    double **images = loadMNISTImages(imgFile, &numImages, &imageSize);
    int numLabels;
    unsigned char *labels = loadMNISTLabels(labelFile, &numLabels);
    if (numImages != numLabels) {
        fprintf(stderr, "Количество изображений и меток не совпадает!\n");
        exit(EXIT_FAILURE);
    }
    *dataSize = numImages;
    TestCase *data = (TestCase*)malloc(numImages * sizeof(TestCase));
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            data[i].image[j] = images[i][j];
        }
        int label = labels[i];
        for (int j = 0; j < OUTPUT_CLASSES; j++) {
            data[i].target[j] = (j == label) ? 1.0 : 0.0;
        }
        free(images[i]);
    }
    free(images);
    free(labels);
    return data;
}

/**
 * @brief  Оценивает сеть и вычисляет усреднённые метрики
 *
 * @param  net       обученная сеть
 * @param  data      массив примеров
 * @param  dataSize  количество примеров
 * @return Metrics   {loss, accuracy, precision, recall}
 */
Metrics evaluateModelMetrics(Network *net, TestCase *data, int dataSize) {
    // инициализируем метрики 0
    Metrics metrics = {0,0,0,0};
    int correct = 0; // число правильных предсказаний
    // массивы для подсчёта TP, FP, FN по каждому классу
    int TP[OUTPUT_CLASSES] = {0}, FP[OUTPUT_CLASSES] = {0}, FN[OUTPUT_CLASSES] = {0};
    double totalLoss = 0;
    for (int i = 0; i < dataSize; i++) {
        // прямой проход
        forwardPropagation(net, data[i].image);
        // считаем Loss
        double sampleLoss = computeLoss(net->layers[net->numLayers - 1].neurons, data[i].target, OUTPUT_CLASSES);
        totalLoss += sampleLoss;
        // определяем предсказанный класс
        int predicted = 0;
        double maxVal = net->layers[net->numLayers - 1].neurons[0];
        for (int j = 1; j < OUTPUT_CLASSES; j++) {
            if (net->layers[net->numLayers - 1].neurons[j] > maxVal) {
                maxVal = net->layers[net->numLayers - 1].neurons[j];
                predicted = j;
            }
        }
        // определяем истинный класс
        int actual = -1;
        for (int j = 0; j < OUTPUT_CLASSES; j++) {
            if (data[i].target[j] == 1.0) {
                actual = j;
                break;
            }
        }
        // обновляем tp, fp, fn
        if (predicted == actual) {
            correct++;
            TP[actual]++;
        } else {
            FP[predicted]++;
            FN[actual]++;
        }
    }
    // усредняем loss
    metrics.loss = totalLoss / dataSize;
    // accuracy = доля правильных предсказаний
    metrics.accuracy = (double)correct / dataSize;
    // cчитаем макро-precision и макро-recall
    double precisionSum = 0, recallSum = 0;
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        double precision = (TP[i] + FP[i] > 0) ? (double)TP[i] / (TP[i] + FP[i]) : 0;
        double recall = (TP[i] + FN[i] > 0) ? (double)TP[i] / (TP[i] + FN[i]) : 0;
        precisionSum += precision;
        recallSum += recall;
    }
    metrics.precision = precisionSum / OUTPUT_CLASSES;
    metrics.recall = recallSum / OUTPUT_CLASSES;
    return metrics;
}

/**
 * @brief  Полный цикл обучения мини-батчами + ранняя остановка
 *
 * @param  net              сеть
 * @param  trainData        обучающая выборка
 * @param  trainSize        её объём
 * @param  valData          валидационная выборка
 * @param  valSize          её объём
 * @param  epochs           максимальное число эпох
 * @param  weightsFile      куда сохранять лучшие веса
 * @param  lossHistoryFile  файл-лог для графика обучения
 */
void trainNetwork(Network *net, TestCase *trainData, int trainSize, TestCase *valData, int valSize, int epochs, const char *weightsFile, const char *lossHistoryFile) {
    // Создание накопителей градиентов
    double ***gradW = (double ***)malloc(net->numLayers * sizeof(double **));
    double **gradB = (double **)malloc(net->numLayers * sizeof(double *));
    for (int l = 1; l < net->numLayers; l++) {
        Layer *curr = &net->layers[l];
        int prevNeurons = net->layerSizes[l-1];
        gradW[l] = (double **)malloc(curr->numNeurons * sizeof(double *));
        gradB[l] = (double *)calloc(curr->numNeurons, sizeof(double));
        for (int i = 0; i < curr->numNeurons; i++) {
            gradW[l][i] = (double *)calloc(prevNeurons, sizeof(double));
        }
    }

    FILE *lossFile = fopen(lossHistoryFile, "w");
    if (!lossFile) {
        perror("Failed to open file for writing loss history");
        exit(EXIT_FAILURE);
    }

    int bestEpoch = 0;
    double bestLoss = 1e9;
    int epochsNoImprove = 0;

    char time_str[9];
    time_t rawtime;
    struct tm * timeinfo;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        if (stopTrainingFlag) {
            printf("Training interrupted by user signal.\n");
            break;
        }
        double epochLoss = 0;
        int numBatches = trainSize / BATCH_SIZE;

        // Перемешивание данных можно добавить здесь

        for (int b = 0; b < numBatches; b++) {
            // Обнуление градиентов
            for (int l = 1; l < net->numLayers; l++) {
                Layer *curr = &net->layers[l];
                int prevNeurons = net->layerSizes[l-1];
                for (int i = 0; i < curr->numNeurons; i++) {
                    memset(gradW[l][i], 0, prevNeurons * sizeof(double));
                }
                memset(gradB[l], 0, curr->numNeurons * sizeof(double));
            }

            double batchLoss = 0;
            int batchStart = b * BATCH_SIZE;
            int currentBatchSize = BATCH_SIZE;
            if(batchStart + currentBatchSize > trainSize)
                currentBatchSize = trainSize - batchStart;

            for (int s = 0; s < currentBatchSize; s++) {
                TestCase *sample = &trainData[batchStart + s];
                forwardPropagation(net, sample->image);
                batchLoss += computeLoss(net->layers[net->numLayers - 1].neurons, sample->target, OUTPUT_CLASSES);
                computeDeltas(net, sample->target);
                for (int l = 1; l < net->numLayers; l++) {
                    Layer *curr = &net->layers[l];
                    Layer *prev = &net->layers[l-1];
                    for (int i = 0; i < curr->numNeurons; i++) {
                        for (int j = 0; j < prev->numNeurons; j++) {
                            gradW[l][i][j] += curr->delta[i] * prev->neurons[j];
                        }
                        gradB[l][i] += curr->delta[i];
                    }
                }
            }
            epochLoss += batchLoss;
            updateWeights(net, gradW, gradB, currentBatchSize);
        }
        epochLoss /= trainSize;

        // Вычисляем метрики на валидационной выборке
        Metrics valMetrics = evaluateModelMetrics(net, valData, valSize);

        // Получаем текущую метку времени
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(time_str, sizeof(time_str), "%H:%M:%S", timeinfo);

        printf("[%s] Epoch %d, Loss=%.6f, Accuracy=%.4f, Precision=%.4f, Recall=%.4f\n",
       time_str, epoch, epochLoss, valMetrics.accuracy, valMetrics.precision, valMetrics.recall);

        fprintf(lossFile, "%d %.6f %.4f %.4f %.4f\n", epoch, epochLoss, valMetrics.accuracy, valMetrics.precision, valMetrics.recall);

        // Механизм ранней остановки (лучше минимизировать потерю)
        if (epochLoss < bestLoss) {
            bestLoss = epochLoss;
            bestEpoch = epoch;
            epochsNoImprove = 0;
            // Сохраняем лучшие веса
            saveWeights(net, weightsFile);
        } else {
            epochsNoImprove++;
        }
        if (epochsNoImprove >= EARLY_STOP_PATIENCE) {
            printf("Early stopping: no improvement for %d epochs.\n", EARLY_STOP_PATIENCE);

            break;
        }
    }

    fclose(lossFile);

    // Освобождение памяти накопителей градиентов
    for (int l = 1; l < net->numLayers; l++) {
        Layer *curr = &net->layers[l];
        int prevNeurons = net->layerSizes[l-1];
        for (int i = 0; i < curr->numNeurons; i++) {
            free(gradW[l][i]);
        }
        free(gradW[l]);
        free(gradB[l]);
    }
    free(gradW);
    free(gradB);

printf("Training completed.\n");
printf("Best result at epoch %d with loss %.6f\n", bestEpoch, bestLoss);
}

/**
 * @brief  Выводит предсказания сети для нескольких примеров
 *
 * @param  net         обученная сеть
 * @param  data        выборка
 * @param  dataSize    её объём
 * @param  numExamples сколько примеров напечатать
 */
void finalEvaluation(Network *net, TestCase *data, int dataSize, int numExamples) {
printf("Final evaluation on %d examples:\n", numExamples);

    if (numExamples > dataSize) numExamples = dataSize;
    for (int i = 0; i < numExamples; i++) {
        forwardPropagation(net, data[i].image);
        printf("Example %d:\n", i+1);
        for (int j = 0; j < OUTPUT_CLASSES; j++) {
            printf("class %d: %.4f ", j, net->layers[net->numLayers - 1].neurons[j]);
        }
        printf("\n");
    }
}


/**
 * @brief  Точка входа программы
 *
 * Поддерживает режимы «train» и «test», выбор датасета
 * (MNIST / синтетический), параметры через CLI.
 *
 * @note   Для прерывания обучения нажмите Ctrl-C —
 *         сработает обработчик сигнала SIGINT.
 */
int main(int argc, char *argv[]) {
    // рвегистрируем обработчик SIGINT для ручной остановки обучения
    signal(SIGINT, handle_sigint);

    // Параметры по умолчанию
    int useMNIST = 1;         // 1 - MNIST, 0 - синтетические данные
    int trainMode = 1;        // 1 - обучаться, 0 - только тест (импорт весов)
    char weightsFile[256] = "weights.txt";
    int epochs = 50;

    // обработка аргументов командной строки
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--dataset=", 10) == 0) {
            if (strstr(argv[i], "synthetic") != NULL)
                useMNIST = 0;
            else if (strstr(argv[i], "mnist") != NULL)
                useMNIST = 1;
        } else if (strncmp(argv[i], "--mode=", 7) == 0) {
            if (strstr(argv[i], "train") != NULL)
                trainMode = 1;
            else if (strstr(argv[i], "test") != NULL)
                trainMode = 0;
        } else if (strncmp(argv[i], "--weights=", 10) == 0) {
            strcpy(weightsFile, argv[i] + 10);
        } else if (strncmp(argv[i], "--epochs=", 9) == 0) {
            epochs = atoi(argv[i] + 9);
        }
    }


    char time_str[9];
    time_t rawtime;
    struct tm * timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(time_str, sizeof(time_str), "%H:%M:%S", timeinfo);
    printf("[%s] Parameters:\n", time_str);
    printf("    Dataset: %s\n", useMNIST ? "MNIST" : "Synthetic");
    printf("    Mode: %s\n", trainMode ? "Train" : "Test");
    printf("    Weights file: %s\n", weightsFile);
    printf("    Epochs: %d\n", epochs);

    srand(time(NULL));

    // чтение конфигурации
    int *layerSizes = NULL;
    int numLayers = 0;
    double learningRate = 0.001, regularization = 0.001;
    readConfig("config.txt", &layerSizes, &numLayers, &learningRate, &regularization);
    printf("Config: ");
    for (int i = 0; i < numLayers; i++) {
        printf("%d ", layerSizes[i]);
    }
    printf("\nlearning_rate: %.6f, regularization: %.6f\n", learningRate, regularization);

    // создание сети
    Network net = createNetwork(layerSizes, numLayers, learningRate, regularization);
    free(layerSizes);

    TestCase *trainData = NULL;
    TestCase *testData = NULL;
    int trainSize = 0, testSize = 0;

    if (useMNIST) {
        printf("Загрузка данных MNIST...\n");
        trainData = loadMNISTData(MNIST_IMAGE_FILE, MNIST_LABEL_FILE, &trainSize);
        // Для валидации можно использовать ту же выборку
        testData = trainData;
        testSize = trainSize;
        printf("Загружено примеров: %d\n", trainSize);
    } else {
        trainSize = SYNTHETIC_SAMPLES;
        trainData = (TestCase*)malloc(trainSize * sizeof(TestCase));
        for (int i = 0; i < trainSize; i++) {
            generateSyntheticData(trainData[i].image, trainData[i].target);
        }
        testSize = SYNTHETIC_TEST_SAMPLES;
        testData = (TestCase*)malloc(testSize * sizeof(TestCase));
        for (int i = 0; i < testSize; i++) {
            generateSyntheticData(testData[i].image, testData[i].target);
        }
        printf("Synth examples for traing: %d, for test: %d\n", trainSize, testSize);
    }

    if (trainMode) {
        // разбиваем данные: можно выделить часть для валидации, здесь используем testData как валидацию
        trainNetwork(&net, trainData, trainSize, testData, testSize, epochs, weightsFile, "loss_history.txt");
    } else {
        // режим только теста (импорт весов)
        loadWeights(&net, weightsFile);
    }

    // финальная проверка на нескольких примерах
    finalEvaluation(&net, testData, testSize, 5);

    // если используются синтетические данные, освобождаем память отдельно
    if (!useMNIST) {
        free(trainData);
        free(testData);
    } else {
        free(trainData);
    }

    freeNetwork(&net);
    return 0;
}
