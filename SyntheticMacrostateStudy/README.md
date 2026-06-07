# SynteticMacrostateStudy

Автономный исследовательский модуль для анализа влияния начальных условий на макросостояния системы взаимодействующих роботов.

Проект объединяет три шага:

1. обучение / загрузку one-step GNN-прогнозера динамики роботов;
2. обучение KMeans-классификатора макросостояний на фактических данных;
3. синтетические multistep-эксперименты на 600 шагов с последующей классификацией состояний системы.

> Название папки `SynteticMacrostateStudy` сохранено в соответствии с текущей структурой проекта.

---

## Цель исследования

Определить, как начальные условия влияют на формирование макросостояний системы роботов, прежде всего на появление и устойчивость мицеллы.

Исследуются факторы:

- начальная плотность / радиус области размещения роботов;
- начальная линейная скорость;
- тип начального направления движения;
- стохастический разброс начальных координат, скоростей и углов.

---

## Основная идея pipeline

```text
fact-data trajectories
        ↓
one-step GNN training
        ↓
autoregressive multistep predictor
        ↓
KMeans macrostate classifier trained on fact-data
        ↓
synthetic initial conditions
        ↓
600-step synthetic rollouts
        ↓
macrostate label for every step
        ↓
micelle onset / lifetime / final state analysis
```

---

## Структура папки

```text
SynteticMacrostateStudy/
├── main.ipynb                         # основной notebook управления экспериментом
├── run_research.py                    # CLI-запуск pipeline
├── config.py                          # центральная конфигурация
├── data_preprocessor.py               # загрузка parquet и расчёт кинематики
├── gnn_model.py                       # GINEConv one-step модель
├── graph_dataset.py                   # подготовка графов для обучения
├── graph_utils.py                     # node/edge features и kNN-графы
├── gnn_trainer.py                     # обучение GNN
├── one_step_trainer.py                # оболочка one-step обучения
├── multistep_predictor.py             # autoregressive multistep rollout
├── initial_state_generator.py         # генератор синтетических начальных условий
├── macrostate_analyzer.py             # macrostate features + KMeans
├── experiment_runner.py               # end-to-end pipeline
├── visualization.py                   # графики и видео
├── requirements.txt
├── data/                              # входные данные
├── models/                            # веса GNN и scaler-файлы
└── results/                           # выходные таблицы, графики, видео, модели
```

---

## Входные данные

### 1. Фактические координаты роботов

Ожидаемый файл:

```text
data/processed_robots_data.parquet
```

Формат:

```text
file_name
slice_id
bot_id
coord_x
coord_y
angle
```

### 2. Большой файл one-step predictions

Файл `gnn_predictions.parquet` слишком большой для хранения в GitHub. Он вынесен в Google Drive:

https://drive.google.com/file/d/1ta-cFK3d8eG9BzpDKPMLvyWPT9Jl2FXE/view?usp=sharing

После скачивания файл нужно положить в путь, который используется проектом для one-step predictions. По умолчанию:

```text
results/gnn_predictions.parquet
```

или в путь, указанный в `config.results_save_path`, если он переопределён.

Если файл отсутствует, проект может заново выполнить one-step inference после обучения модели, но этот этап может быть долгим.

### 3. Веса GNN

По умолчанию:

```text
models/gnn_model_weights.pth
```

Если файла нет, pipeline может обучить one-step GNN заново. Если файл есть и включён режим `skip_training_if_model_exists=True`, обучение будет пропущено.

---

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Для `torch-geometric` может потребоваться установка под конкретную версию PyTorch/CUDA. Если обычная установка не проходит, установите PyG по официальной инструкции для вашей версии PyTorch.

---

## Основной запуск через notebook

Откройте:

```text
main.ipynb
```

и выполняйте ячейки сверху вниз.

Основные этапы notebook:

1. настройка `ResearchConfig`;
2. загрузка fact-data;
3. обучение / загрузка one-step GNN;
4. обучение Macrostate KMeans на фактических данных;
5. multistep evaluation на fact-data;
6. генерация synthetic initial conditions;
7. synthetic study на 600 шагов;
8. сохранение таблиц, графиков и видео.

---

## Ключевая конфигурация последнего эксперимента

В последнем запуске использовались параметры:

```python
data_path = "data/processed_robots_data.parquet"
model_path = "models/gnn_model_weights.pth"
output_dir = "results/research"

# GNN
hidden_dim = 128
gnn_layers = 2
dropout = 0.1
k_neighbors = 5

# One-step training
batch_size = 512
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-4
early_stopping_patience = 10
train_sample_ratio = 0.35

# Multistep / synthetic
start_step = 3
synthetic_total_steps = 600
n_robots = 50
dt = 0.1

# Macrostate
n_clusters = 4
robot_size = 60.0
random_state = 42
```

Сетка synthetic experiments:

```python
center_x = 1000.0
center_y = 500.0

radius_values = [250, 350, 470]
speed_values = [1.0, 3.0, 6.0]
velocity_modes = ["random", "aligned", "rotational", "inward"]
repeats = 2
```

Итого:

```text
3 радиуса × 3 скорости × 4 режима движения × 2 повтора = 72 synthetic experiments
```

---

## Начальные условия synthetic experiments

Идентификатор эксперимента кодирует параметры старта:

```text
synthetic_r350_v1.00_inward_rep0
```

Расшифровка:

```text
r350      → initial radius = 350
v1.00     → speed_mean = 1.00 ед/шаг
inward    → начальные скорости направлены к центру
rep0      → первый случайный повтор этой конфигурации
```

`rep0` и `rep1` отличаются случайной инициализацией координат, углов, скоростей и угловых скоростей при одинаковых глобальных параметрах.

---

## Режимы начального движения

```text
random      → случайные направления скоростей
aligned     → все роботы движутся примерно в одном направлении
rotational  → начальное движение по касательной вокруг центра
inward      → начальное движение к центру области
```

---

## Macrostate KMeans

KMeans обучается только на фактических данных. Синтетические траектории не используются для обучения кластеров.

Для каждого временного шага рассчитываются macrostate-признаки:

```text
Polar_Order
Mean_Angle
Coordination_Num
Mean_Distance
Angular_Vel
Rotation_Direction
Rot_Order
Center_Vel
Velocity_Dispersion
Std_Nearest_Dist
Mean_Velocity
```

Затем:

```text
macrostate features
        ↓
StandardScaler
        ↓
KMeans(n_clusters=4)
        ↓
state label
```

Основные используемые интерпретации кластеров:

```text
Regular Structure (Мицелла)
Sparse Distribution (Разреженное)
```

---

## Multistep prediction

One-step GNN обучается предсказывать:

```text
state(t) → Δx, Δy, Δangle at t+1
```

Multistep rollout работает autoregressive:

```text
fact states 0..3
        ↓
predict step 4
        ↓
recompute velocity/acceleration from predicted history
        ↓
predict step 5
        ↓
...
        ↓
predict step 600
```

После начальной seed-history фактические координаты больше не используются.

---

## Synthetic study outputs

Основные таблицы:

```text
results/research/tables/synthetic_step_states.csv
results/research/tables/synthetic_summary.csv
```

`synthetic_step_states.csv` содержит состояние системы на каждом шаге.

`synthetic_summary.csv` содержит агрегированные показатели по каждому эксперименту:

```text
experiment_id
micelle_formed
first_micelle_step
micelle_total_steps
micelle_total_time
micelle_max_lifetime_steps
micelle_max_lifetime_time
final_state_type
dominant_state_type
radius
speed_mean
velocity_mode
```

Полные synthetic rollout координаты сохраняются в формате fact-data:

```text
results/research/synthetic_trajectories/<experiment_id>.parquet
```

Формат:

```text
file_name
slice_id
bot_id
coord_x
coord_y
angle
```

Это позволяет повторно запускать Macrostate/KMeans-анализ на synthetic trajectories как на обычных фактических данных.

---

## Графики synthetic analysis

Графики сохраняются в:

```text
results/research/plots/synthetic_overview/
```

Ключевые графики:

```text
synthetic_final_state_distribution_by_mode.png
synthetic_mean_first_micelle_step_by_mode.png
synthetic_micelle_probability_by_mode_radius.png
synthetic_micelle_lifetime_radius_speed.png
synthetic_micelle_onset_lifetime_by_velocity_mode.png
synthetic_micelle_onset_lifetime_by_speed.png
synthetic_micelle_onset_lifetime_by_radius.png
```

### Интерпретация onset/lifetime графиков

На графиках:

```text
marker = средний шаг первого появления мицеллы
толстая линия = средняя максимальная непрерывная жизнь мицеллы
прозрачная широкая линия + errorbar = разброс ±1 std
p = вероятность формирования мицеллы
n = число экспериментов с мицеллой / общее число экспериментов в группе
```

Пример:

```text
p=0.83, n=20/24
```

означает, что мицелла появилась в 20 экспериментах из 24.

Важно:

```text
micelle_formed=True
```

означает, что мицелла появилась хотя бы один раз за 600 шагов.

```text
final_state_type
```

показывает только состояние на последнем шаге rollout. Поэтому возможна ситуация, когда мицелла появилась, но к концу симуляции система перешла в разреженное состояние.

---

## Результаты последней synthetic simulation

В последнем запуске:

```text
Всего synthetic experiments: 72
Мицелла появилась хотя бы один раз: 68 / 72 = 94.4%
Финальное состояние = Мицелла: 23 / 72 = 31.9%
Финальное состояние = Разреженное: 49 / 72 = 68.1%
Доминирующее состояние = Мицелла: 52 / 72 = 72.2%
```

### Влияние начального радиуса

```text
radius=250: p=1.00, first step≈0.0, max lifetime≈322 steps, final micelle≈54.2%
radius=350: p=1.00, first step≈0.0, max lifetime≈256 steps, final micelle≈25.0%
radius=470: p=0.83, first step≈53.0, max lifetime≈205 steps, final micelle≈16.7%
```

Вывод: чем плотнее стартовая конфигурация, тем быстрее формируется и дольше сохраняется мицелла.

### Влияние начальной скорости

```text
speed=1: p=1.00, first step≈25.6, max lifetime≈267 steps
speed=3: p=0.92, first step≈13.4, max lifetime≈231 steps
speed=6: p=0.92, first step≈6.9,  max lifetime≈295 steps
```

Вывод: увеличение скорости ускоряет появление мицеллы; при speed=6 наблюдается наиболее ранний onset и высокая устойчивость.

### Влияние режима начального движения

```text
inward:     p=1.00, first step≈6.2,  max lifetime≈337 steps, final micelle≈50.0%
aligned:    p=1.00, first step≈11.1, max lifetime≈209 steps, final micelle≈16.7%
random:     p=0.89, first step≈10.1, max lifetime≈246 steps, final micelle≈33.3%
rotational: p=0.89, first step≈36.8, max lifetime≈264 steps, final micelle≈27.8%
```

Вывод: движение к центру (`inward`) наиболее благоприятно для быстрой и устойчивой мицеллы. Вращательный режим (`rotational`) формирует мицеллу позднее.

---

## Видео

Видео сохраняются в:

```text
results/research/videos/
```

Полезные функции:

```python
pipeline.save_one_step_video(file_name=demo_file)
pipeline.save_multistep_video(predictions_df=multistep_predictions, file_name=demo_file)
pipeline.save_synthetic_video(experiment_id="synthetic_r350_v1.00_inward_rep0")
```

Synthetic video показывает не только движение роботов, но и KMeans-состояние на кадре:

```text
step
cluster
state_type
micelle=True/False
```

---

## Быстрые команды

Полный запуск:

```bash
python run_research.py --full
```

Полный запуск с переобучением GNN:

```bash
python run_research.py --full --force-retrain
```

Только one-step training:

```bash
python run_research.py --train-onestep
```

Только KMeans:

```bash
python run_research.py --fit-macrostate
```

Только synthetic study:

```bash
python run_research.py --run-synthetic
```

---

## Ограничения

1. One-step GNN не является аналитической кинематической моделью. Она чувствительна к распределению координат, на котором обучалась.
2. Synthetic initial conditions должны быть заданы в той же системе координат, что и fact-data.
3. Multistep rollout накапливает ошибку, так как каждый следующий шаг зависит от предыдущего прогноза.
4. KMeans-кластеры являются data-driven состояниями и требуют физической интерпретации.
5. Для более устойчивой статистики synthetic study желательно увеличить `repeats`.

---

## Рекомендуемые следующие шаги

- увеличить число repeats для synthetic grid;
- добавить больше значений `radius` и `speed_mean`;
- провести synthetic study с непрерывной выборкой скоростей;
- проверить выводы synthetic study на фактических экспериментах;
- обучить GNN с более инвариантными признаками: относительные координаты, нормировка относительно центра, boundary-aware features.
