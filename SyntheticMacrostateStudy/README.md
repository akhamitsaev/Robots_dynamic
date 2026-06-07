# Robots Dynamic: автономный пакет обучения one-step GNN, multistep-прогноза и исследования мицеллы

Пакет реализует конечную цель исследования без ручного переноса модели из `StepPrediction`:

1. Загружает фактические данные `processed_robots_data.parquet`.
2. Обучает one-step GNN внутри этой же папки.
3. Сохраняет веса в `models/gnn_model_weights.pth`.
4. Использует эти веса для autoregressive multistep-прогноза.
5. Обучает KMeans macrostate-классификатор на фактических данных.
6. Генерирует синтетические начальные условия.
7. Запускает synthetic rollout длиной 600 шагов.
8. Классифицирует состояние системы на каждом шаге и считает факт/шаг/длительность мицеллы.

## Структура

```text
Robots_dynamic_multistep_research/
├── main.ipynb                         # главный notebook управления всем пайплайном
├── run_research.py                    # CLI запуск
├── config.py                          # центральный конфиг
├── data_preprocessor.py               # загрузка parquet + кинематика
├── gnn_model.py                       # простая GINEConv one-step модель
├── graph_utils.py                     # node/edge features и kNN граф
├── one_step_trainer.py                # обучение one-step GNN внутри пакета
├── multistep_predictor.py             # autoregressive rollout + multistep metrics
├── initial_state_generator.py         # генератор начальных условий
├── macrostate_analyzer.py             # 11 macrostate features + KMeans classifier
├── experiment_runner.py               # end-to-end pipeline
├── visualization.py                   # графики и видео/анимации real vs predicted
├── data/                              # сюда положить processed_robots_data.parquet
├── models/                            # сюда автоматически сохраняются веса GNN
└── results/                           # выходные таблицы/графики/видео/модели
```

## Что нужно положить вручную

Теперь нужен только файл фактических данных:

```text
StepPrediction/data/processed_robots_data.parquet -> data/processed_robots_data.parquet
```

Веса GNN переносить не нужно: они обучаются автономно и сохраняются здесь:

```text
models/gnn_model_weights.pth
```

Если файл уже существует, по умолчанию обучение пропускается. Для принудительного переобучения используйте `force_retrain=True` в notebook или `--force-retrain` в CLI.

## Установка

```bash
pip install -r requirements.txt
```

PyTorch Geometric иногда требует установку под конкретную версию CUDA/PyTorch. Если обычный `pip install torch-geometric` не подходит, установите PyG по инструкции для вашей версии PyTorch/CUDA.

## Главная логика

### One-step обучение

Для каждого фактического эксперимента строятся графы по временным срезам:

```text
state(t), graph(t), node_features(t), edge_features(t) -> delta_state(t -> t+1)
```

Используются те же признаки, что и в текущем проекте:

- node features: 9 признаков на робота;
- edge features: 11 признаков на ребро;
- kNN-граф на каждом временном срезе;
- target: `Δx, Δy, Δangle` для каждого робота.

Обученная модель сохраняется в `config.model_path`.

Объём fact-файлов для обучения регулируется параметром:

```python
train_sample_ratio = 1.0  # 1.0 = все fact-файлы, 0.5 = случайные 50% файлов
```

Дополнительно можно ограничить число временных графов из каждого файла:

```python
max_train_samples_per_file = None  # или, например, 50 для быстрого debug
max_val_samples_per_file = None
max_test_samples_per_file = None
```

### Multistep прогноз

Multistep не использует фактический текущий шаг после старта:

```text
seed states 0..3 -> predict step 4
predicted 0..4  -> predict step 5
predicted 0..5  -> predict step 6
...
```

Скорости и ускорения на каждом новом шаге пересчитываются из уже предсказанной истории.

### Macrostate

KMeans обучается только на фактических macrostate-признаках. Синтетические траектории не участвуют в обучении KMeans, они только классифицируются уже обученной моделью.

## Запуск через notebook

Откройте:

```text
main.ipynb
```

и выполните ячейки сверху вниз.

В notebook есть отдельные этапы:

1. конфигурация;
2. загрузка fact-data;
3. обучение one-step GNN;
4. обучение Macrostate KMeans;
5. multistep evaluation на fact-data;
6. synthetic study 600 steps.

## CLI запуск

Полный автономный запуск:

```bash
python run_research.py --full
```

Полный запуск с принудительным переобучением one-step GNN:

```bash
python run_research.py --full --force-retrain
```

Пример с управлением синтетической скоростью и геометрией:

```bash
python run_research.py --full --speed-min 5 --speed-max 40 --n-speed-samples 3 --dish-diameter 1000 --robot-size 60
```

Только обучение one-step GNN:

```bash
python run_research.py --train-onestep
```

Только обучение Macrostate KMeans:

```bash
python run_research.py --fit-macrostate
```

Только multistep evaluation, если модель уже обучена:

```bash
python run_research.py --evaluate-real
```

Только synthetic study, если уже есть `models/gnn_model_weights.pth` и `results/research/models/macrostate_kmeans.pkl`:

```bash
python run_research.py --run-synthetic
```


## Synthetic initial conditions: скорость и геометрия

В обновлённой версии `speed_values` больше не задаётся фиксированным списком `[10, 20, 35]`.
Если в `InitialStateGenerator.make_sweep_specs(...)` передать `speed_values=None`, то генератор сэмплирует непрерывные значения скорости из диапазона:

```python
speed_range=(config.speed_min, config.speed_max)
n_speed_samples=config.n_speed_samples  # по умолчанию 3
```

По умолчанию это 3 случайных непрерывных значения из диапазона `[10, 35]`. Они сэмплируются один раз и затем используются во всех комбинациях `radius_values × velocity_modes × repeats`, чтобы сетка экспериментов оставалась интерпретируемой.

Геометрия синтетического старта теперь учитывает физический размер роботов:

```python
dish_diameter = 1000.0
robot_size = 60.0  # диаметр робота
min_center_distance = None  # None => robot_size
```

Координаты считаются координатами центров роботов. Поэтому для тарелки диаметром 1000 максимальный радиус размещения центров равен:

```text
500 - 60/2 = 470
```

Генератор не допускает начальные положения, где расстояние между центрами роботов меньше 60.

Рекомендуемые `radius_values` для 50 роботов в тарелке диаметром 1000:

```python
radius_values = [250, 350, 470]
```

Интерпретация:

```text
250 — плотный старт, area fraction ≈ 0.72
350 — средняя плотность, area fraction ≈ 0.37
470 — разреженный старт почти на всю доступную тарелку, area fraction ≈ 0.20
```

Area fraction здесь примерно считается как:

```text
phi = n_robots * (robot_diameter / 2)^2 / radius^2
```

## Что сохраняется

```text
models/gnn_model_weights.pth
results/research/tables/one_step_training_history.csv
results/research/tables/one_step_test_metrics.csv
results/research/tables/one_step_test_predictions.parquet
results/research/tables/one_step_split.json
results/research/tables/fact_cluster_results.csv
results/research/tables/fact_state_interpretation.csv
results/research/tables/multistep_fact_predictions.parquet
results/research/tables/multistep_fact_metrics.csv
results/research/tables/synthetic_step_states.csv
results/research/tables/synthetic_summary.csv
results/research/plots/*.png
results/research/models/macrostate_kmeans.pkl
```

## Основные параметры в `config.py`

```python
batch_size = 512
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-4
early_stopping_patience = 10
hidden_dim = 128
gnn_layers = 2
k_neighbors = 5
start_step = 3
synthetic_total_steps = 600
n_robots = 50
robot_size = 60.0
dish_diameter = 1000.0
speed_min = 10.0
speed_max = 35.0
n_speed_samples = 3
n_clusters = 4

# выборка для one-step обучения
train_sample_ratio = 1.0
max_train_samples_per_file = None

# видео/анимации
save_animations = True
animation_fps = 10
animation_max_frames = 300
```

Для быстрого smoke-test можно временно поставить:

```python
num_epochs = 2
train_sample_ratio = 0.2
max_train_samples_per_file = 50
max_val_samples_per_file = 20
max_test_samples_per_file = 20
```

## Видео движения роботов

Видео создаётся в `results/research/videos/`:

```text
results/research/videos/one_step_real_vs_predicted.mp4      # или .gif, если ffmpeg недоступен
results/research/videos/multistep_real_vs_predicted_*.mp4   # или .gif
```

Реализация использует тот же подход, что и исходный `StepPrediction/main.ipynb`: `matplotlib.animation.FuncAnimation` + `matplotlib.patches.Ellipse`, реальные роботы синие, предсказанные красные. В исходном notebook сохранение было через `writer="ffmpeg"`; в автономном пакете добавлен Windows-safe fallback: если на Windows 11 нет доступного `ffmpeg`, автоматически сохраняется `.gif` через Pillow.

Управление:

```python
save_animations = True
animation_fps = 10
animation_max_frames = 300  # None — все кадры
animation_robot_width = 60.0
animation_robot_height = 36.0
```

В CLI:

```bash
python run_research.py --full --no-animations
python run_research.py --full --animation-max-frames -1
```

## Важное замечание о нормализации

По умолчанию `standardize_gnn=False`, чтобы one-step training и multistep inference работали без дополнительных файлов. Если включить:

```python
standardize_gnn=True
```

то trainer сохранит scaler-файлы рядом с моделью:

```text
models/gnn_model_weights_node_scaler.pkl
models/gnn_model_weights_edge_scaler.pkl
models/gnn_model_weights_target_scaler.pkl
```

`MultiStepPredictor` автоматически загрузит их и применит при multistep rollout.


## Видео после обучения, без переобучения

В версии v5 видео создаётся отдельным шагом после обучения/evaluation. По умолчанию:

```python
config.auto_save_animations_during_pipeline = False
```

Поэтому `pipeline.train_one_step_model(...)` только обучает модель и сохраняет предсказания, а видео создаётся отдельно:

```python
one_step_video_path = pipeline.save_one_step_video()
```

Для multistep сначала выполняется evaluation:

```python
multistep_predictions, multistep_metrics = pipeline.evaluate_multistep_on_fact_data(experiments, horizon=200)
```

Затем видео сохраняется отдельной командой:

```python
multistep_video_path = pipeline.save_multistep_video(predictions_df=multistep_predictions)
```

CLI без переобучения:

```bash
python run_research.py --save-onestep-video
python run_research.py --save-multistep-video
```

Если нужно старое поведение, когда видео создаётся автоматически внутри pipeline, установите:

```python
config.auto_save_animations_during_pipeline = True
```


## One-step GNN compatibility note

The one-step GNN training path uses the original StepPrediction-style normalization: `StandardScaler` is fitted on train node features, edge features, and targets inside `gnn_trainer.py`. This behavior is always enabled to match the GitHub one-step pipeline. There is no `standardize_gnn` switch in this version.

### Почему после обучения появляется "Прогнозирование для всех данных..."

После train/test оригинальный GitHub-style `GNNTrainer` дополнительно делает full one-step inference по всем переданным fact-файлам и сохраняет `results/gnn_predictions.parquet`. Это нужно для one-step таблицы результатов и последующего видео. На больших данных этот этап может быть долгим: например, 43 451 граф × около 50 роботов = больше 2 млн строк predictions.

В v9 добавлен progress bar для этого этапа. Если нужно только обучить веса и не сохранять one-step predictions сразу, можно временно поставить в `config.py` или `main.ipynb`:

```python
config.save_one_step_predictions_after_training = False
```

Для GitHub-compatible поведения и для one-step video оставляйте значение по умолчанию:

```python
config.save_one_step_predictions_after_training = True
```

## Demo experiment selection for videos (GitHub notebook logic)

The original `StepPrediction/main.ipynb` selected a concrete experiment before plotting/video export, e.g.:

```python
sub_df = df[df['file_name'] == test_files[2]]
```

This package keeps the same logic. The demo experiment is controlled by:

```python
config.demo_split = "test"
config.demo_file_index = 2
config.demo_file_name = None  # set this to an exact file_name to override index selection
```

In `main.ipynb`:

```python
demo_file = pipeline.select_demo_file(experiments)  # default: test_files[2]

one_step_video_path = pipeline.save_one_step_video(file_name=demo_file)
multistep_video_path = pipeline.save_multistep_video(
    predictions_df=multistep_predictions,
    file_name=demo_file,
)
```

If `results/gnn_predictions.parquet` does not exist, `save_one_step_video(...)` will not retrain the model. It will run one-step teacher-forcing inference only for `demo_file` using the existing trained checkpoint and scaler files.

CLI equivalents:

```bash
python run_research.py --save-onestep-video --demo-file-index 2 --demo-split test
python run_research.py --save-multistep-video --demo-file-index 2 --demo-split test
```

or with an exact experiment name:

```bash
python run_research.py --save-onestep-video --demo-file-name "01_69_[45_bots_PWM_2_ex_108].pickle"
```
