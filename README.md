# Robots Dynamic: прогнозирование динамики и анализ макросостояний системы роботов

Репозиторий посвящён моделированию, прогнозированию и анализу коллективной динамики системы взаимодействующих роботов.

Главная исследовательская задача:

```text
определить влияние начальных условий на макросостояние системы
и спрогнозировать динамику взаимодействующих роботов.
```

Особый фокус — формирование и устойчивость мицеллы как коллективного состояния системы.

---

## Структура проекта

Проект логически разделён на три папки:

```text
Robots_dynamic/
├── StepPrediction/              # one-step прогноз координат и угла роботов
├── Macrostate/                  # KMeans-классификация макросостояний
└── SynteticMacrostateStudy/     # multistep + synthetic study начальных условий
```

---

## 1. StepPrediction

### Назначение

Папка `StepPrediction` реализует one-step прогноз состояния каждого робота:

```text
x(t), y(t), angle(t) → x(t+1), y(t+1), angle(t+1)
```

Сравниваются два подхода:

1. кинематическая модель;
2. GNN-модель на графах взаимодействий роботов.

### Кинематический baseline

Baseline использует скорости и ускорения:

```text
x_pred(t) = x(t-1) + Vx(t-1)·dt + 0.5·ax(t-1)·dt²
y_pred(t) = y(t-1) + Vy(t-1)·dt + 0.5·ay(t-1)·dt²
θ_pred(t) = θ(t-1) + ω(t-1)·dt + 0.5·α(t-1)·dt²
```

### GNN model

На каждый временной срез строится граф:

```text
nodes = robots
edges = k nearest neighbors
```

Node features:

```text
x, y, angle, vx, vy, angular_velocity, ax, ay, angular_acceleration
```

Edge features:

```text
Δx, Δy, distance, relative_angle, Δangle,
Δvx, Δvy, Δax, Δay, Δangular_velocity, Δangular_acceleration
```

Модель:

```text
node/edge encoders
        ↓
GINEConv × 2
        ↓
BatchNorm + ReLU + Dropout
        ↓
predictor head
        ↓
Δx, Δy, Δangle
```

### Большой файл predictions

Файл `gnn_predictions.parquet` содержит one-step прогнозы координат и слишком большой для GitHub.

Он сохранён отдельно:

https://drive.google.com/file/d/1ta-cFK3d8eG9BzpDKPMLvyWPT9Jl2FXE/view?usp=sharing

Рекомендуемое расположение после скачивания:

```text
StepPrediction/results/gnn_predictions.parquet
```

или путь, указанный в конфиге соответствующего модуля.

---

## 2. Macrostate

### Назначение

Папка `Macrostate` переводит траекторию отдельных роботов в макросостояние всей системы.

Идея:

```text
coordinates over time
        ↓
macrostate feature extraction
        ↓
StandardScaler
        ↓
KMeans clustering
        ↓
state label per time step
```

### Macrostate features

Используются признаки коллективной динамики:

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

### Интерпретация

Кластеры KMeans интерпретируются как состояния системы. В текущем исследовании ключевые состояния:

```text
Regular Structure (Мицелла)
Sparse Distribution (Разреженное)
```

KMeans обучается только на фактических данных. Прогнозные и синтетические траектории затем классифицируются уже обученной моделью.

---

## 3. SynteticMacrostateStudy

### Назначение

Папка `SynteticMacrostateStudy` объединяет one-step GNN, multistep rollout и Macrostate KMeans для исследования влияния начальных условий.

Pipeline:

```text
fact-data
   ↓
train/load one-step GNN
   ↓
train/load KMeans macrostate classifier
   ↓
generate synthetic initial states
   ↓
600-step autoregressive rollout
   ↓
classify macrostate at every step
   ↓
estimate micelle formation and lifetime
```

### Synthetic grid

Последняя конфигурация synthetic study:

```text
n_robots = 50
dt = 0.1
total_steps = 600
center = (1000, 500)
radius_values = [250, 350, 470]
speed_values = [1.0, 3.0, 6.0]
velocity_modes = random, aligned, rotational, inward
repeats = 2
```

Итого:

```text
3 × 3 × 4 × 2 = 72 synthetic experiments
```

### Synthetic outputs

```text
results/research/tables/synthetic_step_states.csv
results/research/tables/synthetic_summary.csv
results/research/synthetic_trajectories/<experiment_id>.parquet
results/research/plots/synthetic_overview/*.png
results/research/videos/*.mp4
```

Synthetic trajectories сохраняются в том же формате, что и fact-data:

```text
file_name, slice_id, bot_id, coord_x, coord_y, angle
```

---

## Основные результаты synthetic study

В последнем synthetic запуске:

```text
Всего экспериментов: 72
Мицелла появилась хотя бы один раз: 68 / 72 = 94.4%
Финальное состояние = Мицелла: 23 / 72 = 31.9%
Финальное состояние = Разреженное: 49 / 72 = 68.1%
Доминирующее состояние = Мицелла: 52 / 72 = 72.2%
```

### Главные выводы

1. Начальная плотность является главным фактором формирования мицеллы.
2. При `radius=250` и `radius=350` мицелла появляется во всех synthetic experiments.
3. При `radius=470` вероятность формирования падает до `0.83`.
4. Большая начальная скорость ускоряет появление мицеллы.
5. Режим `inward` даёт наиболее быстрое и устойчивое формирование мицеллы.
6. Финальное состояние и факт появления мицеллы — разные метрики: система может сформировать мицеллу, а затем перейти в разреженное состояние.

---

## Как запускать проект целиком

Рекомендуемый порядок:

```text
1. Подготовить data/processed_robots_data.parquet.
2. Обучить one-step GNN в StepPrediction или внутри SynteticMacrostateStudy.
3. Обучить KMeans в Macrostate или внутри SynteticMacrostateStudy.
4. Запустить synthetic study.
5. Проанализировать synthetic_summary.csv и графики.
```

Для автономного запуска лучше использовать `SynteticMacrostateStudy`, так как он содержит полный end-to-end pipeline.

---

## Данные, которые не хранятся в GitHub

Из-за размера не хранится:

```text
gnn_predictions.parquet
```

Ссылка:

https://drive.google.com/file/d/1ta-cFK3d8eG9BzpDKPMLvyWPT9Jl2FXE/view?usp=sharing

После скачивания файл нужно положить в `results/` соответствующего модуля или в путь, заданный в конфиге.

---

## Научная интерпретация

Проект позволяет перейти от локального прогноза координат роботов к анализу коллективных режимов системы.

```text
GNN отвечает на вопрос:
где будет каждый робот на следующем шаге?

KMeans macrostate model отвечает на вопрос:
в каком коллективном состоянии находится система?

Synthetic study отвечает на вопрос:
какие начальные условия приводят к формированию и сохранению мицеллы?
```

---

## Ограничения

- GNN обучена на распределении фактических координат, поэтому synthetic initial conditions должны быть заданы в той же системе координат.
- Multistep rollout накапливает ошибку.
- KMeans-кластеры являются data-driven и требуют физической интерпретации.
- Для более надёжной статистики нужно увеличить число повторов synthetic experiments.

---

## Рекомендуемое развитие

- увеличить число synthetic repeats;
- расширить сетку `radius` и `speed_mean`;
- добавить анализ непрерывного распределения скоростей;
- валидировать synthetic conclusions на fact-data;
- обучить GNN с относительными и boundary-aware признаками;
- добавить более физически интерпретируемые macrostate labels.
