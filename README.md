 ## 📂 Структура проекта

```

.
├── build.ninja           # Генерируемый Ninja-скрипт
├── CMakeCache.txt        # Кэш CMake
├── CMakeFiles/           # Артефакты CMake
├── cmake\_install.cmake   # Скрипт установки
├── config.txt            # Конфигурация сети (число нейронов, lr, L2-рег.)
├── loss\_history.txt      # История потерь/метрик при обучении
├── NeuralNetwork         # Исполняемый файл тренировки/теста
├── mnist_gui                # Исполняемый GUI для ручного теста
└── weights.txt           # Сохранённые веса сети

```

- **NeuralNetwork** — главный исполняемый файл:  
  - `main.c` – реализация обучения, теста, сохранения/загрузки весов  
- **mnist_gui** — GUI приложение в котором :  
  - `mnist_gui.c` – читает `config.txt`, загружает `weights.txt` и позволяет нарисовать цифру мышкой выводя в реальном времени вероятности того какая цифра была нарисована.
- **config.txt** — задайте в нём, например:  
```

neurons: 784,128,64,10
learning\_rate: 0.001
regularization: 0.001

````
- **weights.txt** — (по умолчанию) файл для сохранения/загрузки весов  
- **loss_history.txt** — логи по эпохам (loss, accuracy, precision, recall)

## 🔧 Зависимости

- CMake ≥ 3.10  
- Ninja  
- GCC или Clang с поддержкой C99  
- POSIX-совместимая ОС (Linux/Mac)

## 🏗 Сборка

```bash

# Компиляция main.c → cmake-build-debug/NeuralNetwork
gcc main.c -o cmake-build-debug/NeuralNetwork.exe -lm

# Компиляция mnist_gui.c → cmake-build-debug/mnist_gui 
gcc mnist_gui.c -lgdi32 -o cmake-build-debug/mnist_gui.exe

````

После успешной сборки в корне появятся два исполняемых файла: `NeuralNetwork` и `mnist_gui`.

## ⚙ Конфигурация

Прежде чем запускать, убедитесь, что в `config.txt` указаны нужные вам размеры слоёв и гиперпараметры.
Если хотите использовать свои файлы MNIST, отредактируйте пути в макросах `MNIST_IMAGE_FILE` и `MNIST_LABEL_FILE` в `main.c`.

## ▶️ Запуск

### Обучение / Тестирование сети

```bash
./NeuralNetwork [--dataset=mnist|synthetic] [--mode=train|test] [--weights=FILE] [--epochs=N]
```

* `--dataset`: `mnist` (по умолчанию) или `synthetic`
* `--mode`: `train` (обучение, по умолчанию) или `test` (только загрузка весов и финальная проверка)
* `--weights`: путь к файлу весов (по умолчанию `weights.txt`)
* `--epochs`: число эпох (по умолчанию `50`)

**Примеры:**

```bash
# Обучить на MNIST 30 эпох, сохранить в default weights.txt
./NeuralNetwork --epochs=30

# Обучить на синтетических данных, 100 эпох
./NeuralNetwork --dataset=synthetic --epochs=100

# Только тест с загрузкой весов из my_weights.txt
./NeuralNetwork --mode=test --weights=my_weights.txt
```

В процессе обучения:

* Лучшие веса автоматически сохраняются в указанный файл.
* Логи метрик (loss, accuracy, precision, recall) сохраняются в `loss_history.txt`.
