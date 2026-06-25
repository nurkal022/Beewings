# BeeWings

Полный программный стек для морфометрии крыльев медоносной пчелы: ручная разметка, две независимые нейросетевые модели (Алпатов 12, Тофильский 19), автоматический расчёт классических индексов (CI, DsA, RI), пакетная обработка.

См. также: [METHODOLOGY.md](METHODOLOGY.md) — подробная научная методология.

---

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Быстрый старт (демо)

```bash
beewings
```

Откроется **стартовый экран**: «Открыть папку со сканами…» → выбрать папку проекта
(несколько больших сканов). Откроется окно проекта с вкладками:
**Нарезка** (авто-нарезка + правка рамок) → **Точки** (ML + ручная правка) →
**Экспорт** (папки + таблица + TPS + отчёт). Недавние проекты доступны на
стартовом экране.

---

## Две независимые методики

| Профиль | Точки | Методика | Checkpoint | Используется для |
|---|---|---|---|---|
| **Алпатов 8 точек** | 8 | Линейная (Россия, 1948) | `alpatov12.pt` | Кубитальный индекс CI |
| **Алпатов 12 точек** | 12 | Линейная | `alpatov12.pt` | CI, DsA, RI, PI индексы |
| **Тофильский 19 точек** | 19 | Геометрическая (Польша, 2008+) | `tofilski19.pt` | Procrustes / IdentiFly / DeepWings совместимая разметка |

Методики **строго разделены**: разные модели, разные ID точек, разные формулы индексов. Загрузка чужого checkpoint'а под профиль блокируется с ошибкой `Несовпадение схемы`.

---

## Точность моделей

| | Tofilski 19pt | Alpatov 12pt |
|---|---|---|
| **val median** | **1.55 px** | **1.53 px** |
| **val mean** | 1.81 px | 1.67 px |
| **val p90** | 2.61 px | 2.57 px |
| Размер train | 6283 крыла | 5904 крыла |

---

## CLI

| Команда | Назначение |
|---|---|
| `beewings` | GUI разметка + ML авто-определение |
| `beewings-ml-prepare` | TPS-дерево → единый CSV для тренировки |
| `beewings-ml-train` | Тренировка UNet на CSV |
| `beewings-ml-eval` | Per-landmark + overall pixel error |
| `beewings-ml-predict` | Предсказание на одном изображении (JSON в stdout) |
| `beewings-ml-batch` | Пакетная обработка папки → JSON-аннотации + CSV + индексы |
| `beewings-crop` | Авто-нарезка скана на отдельные крылья (контурная сегментация) |
| `beewings-pipeline` | Окно-мастер: сканы → нарезка → правка → точки → экспорт |
| `beewings-ml-report` | Paper-ready графики (per-landmark error, hist, sample overlays) |
| `beewings-detect` | Классический CV-детектор (резерв без обучения) |
| `beewings-eval` | Eval классического детектора |
| `beewings-canonicalize` | Каноникализация порядка точек в CSV |
| `beewings-api` | HTTP REST API (FastAPI) — детекция / нарезка / индексы / экспорт |

### Пакетный пример

```bash
# обработать все .jpg в demo_wings/, сохранить CSV+индексы+JSON
beewings-ml-batch \
    --checkpoint checkpoints/alpatov12.pt \
    --images demo_wings \
    --csv predictions.csv \
    --report report.json \
    --tta \
    --include-confidence
```

В `report.json` для каждого изображения окажется блок с уверенностью точек и значениями индексов (CI Алпатова, DsA, RI, ...).

### Авто-нарезка скана

Большой скан стекла (этикетка слева + сетка из 20–50 крыльев) автоматически
режется на отдельные изображения крыльев, готовые для `beewings-ml-batch`.

```bash
# нарезать один скан на отдельные крылья (+ debug-оверлей для проверки)
beewings-crop --input "скан/Клат/8.jpg" --out crops/Клат --debug

# вся папка рекурсивно, с поворотом крыльев в горизонталь для ML
beewings-crop --input "скан/" --out crops/ --recursive --rotate-canonical
```

Выход: `<скан>_crop_0.jpg … _crop_N.jpg` (отдельные крылья в порядке чтения),
`<скан>_label.jpg` (этикетка), `<скан>_debug.jpg` (оверлей с пронумерованными
рамками и красной рамкой этикетки — для визуальной проверки нарезки).

Ключи: `--inplace` (писать рядом со сканом), `--margin` (поля вокруг крыла, по
умолчанию 0.10), `--min-area-frac` (порог площади крыла), `--rotate-canonical`
(повернуть в горизонталь, основание слева — рекомендуется для ML), `--dry-run`
(только оверлей, без кропов). Проверено на реальных сканах: 28/50/24 крыла без
захвата этикетки.

Этикетка слева определяется автоматически. Если раскладка диагональная и
автоопределение не сработало (этикетка делит колонки с крыльями), задайте её
вручную:

```bash
# замаскировать левые 27% ширины как этикетку
beewings-crop --input "скан/Ип Красон/03201.jpg" --out crops/Красон --label-frac 0.27

# либо точная рамка в пикселях X,Y,W,H
beewings-crop --input scan.jpg --out crops/ --label-box 0,0,3400,6642
```

### Конвейер в окне (мастер)

Полный рабочий процесс одним мастером — в меню **«🧩 Конвейер нарезки…»**
(или команда `beewings-pipeline`):

1. **Сканы** — выбрать папку с большими сканами.
2. **Нарезка и правка** — авто-нарезка всех; на большом скане можно удалять
   лишние рамки (ПКМ), рисовать недостающие (протяжка ЛКМ), тянуть за углы;
   «Пересоздать кропы» сохраняет крылья.
3. **Точки** — выбрать профиль и «Расставить точки» (ML), затем «Открыть в
   редакторе точек» для ручной правки.
4. **Экспорт** — структура папок + сводная таблица (CSV+индексы) + TPS +
   сводный отчёт в `<проект>/export/`.

Проект (`project.json`) сохраняется в рабочей папке, процесс можно продолжить.

### Paper-ready отчёт

```bash
beewings-ml-report \
    --csv data/wings12.csv \
    --checkpoint checkpoints/alpatov12.pt \
    --image-root data/wings12_images \
    --out reports/alpatov12_test \
    --split test --tta
```

Результат в `reports/alpatov12_test/`:
- `summary.json` — машинно-читаемый JSON для статьи
- `eval_table.csv` — таблица per-landmark
- `per_landmark.png` — bar chart mean/median/p90 ошибок по точкам
- `error_hist.png` — гистограмма распределения ошибок
- `sample_overlays/` — 12 примеров с GT (зелёный) vs prediction (красный)

---

## HTTP API

Тонкая REST-обёртка (FastAPI) над теми же функциями ядра — чтобы работать с
системой программно. Все эндпоинты принимают **и** загрузку файла (multipart),
**и** путь к файлу на сервере.

### Запуск

```bash
pip install -e ".[api]"
beewings-api                       # http://0.0.0.0:8000 (Swagger UI на /docs)
# или: uvicorn beewings.api.app:app --reload
```

Переменные окружения: `BEEWINGS_API_HOST` / `BEEWINGS_API_PORT`,
`BEEWINGS_CHECKPOINTS` (папка с `.pt`, по умолчанию `checkpoints/`),
`BEEWINGS_DEVICE` (`auto`/`cuda`/`mps`/`cpu`),
`BEEWINGS_DATA_ROOT` (ограничивает серверные пути для безопасности),
`BEEWINGS_CORS_ORIGINS`.

### Эндпоинты

| Метод/путь | Назначение |
|---|---|
| `GET /health` | Статус, устройство, загруженные модели |
| `GET /models` | Доступные методики и чекпоинты |
| `POST /detect` | Картинка крыла → точки + confidence (+ индексы для Алпатов-12) |
| `POST /segment` | Скан → отдельные крылья (`output=inline` base64 / `output=files`) |
| `POST /indices` | Координаты 12 точек → CI/DsA/RI + кандидаты подвида |
| `POST /export` | Аннотации → TPS/XLSX/JSON (`format=tps\|xlsx\|json\|all`) |

```bash
# Детекция (Алпатов 12 точек), загрузка файла, с TTA
curl -F image=@demo_wings/demo_realdata_01.jpg \
     -F methodology=alpatov -F tta=true \
     localhost:8000/detect

# Детекция по серверному пути (JSON-режим)
curl -X POST localhost:8000/detect -H 'content-type: application/json' \
     -d '{"image_path":"/data/wing.jpg","methodology":"tofilski"}'

# Нарезка скана, кропы в ответе как base64
curl -F scan=@demo_scans/01_klat.jpg "localhost:8000/segment?" -F output=inline

# Экспорт в zip (tps+xlsx+json)
curl -X POST "localhost:8000/export?format=all" -H 'content-type: application/json' \
     -d '{"methodology":"alpatov","wings":[{"wing":"c0.jpg","landmarks":[{"id":1,"x":10,"y":5}]}]}' \
     -o export.zip
```

> Индексы (CI и др.) помечены как **предварительные** — ID точек должен подтвердить
> биолог по эталонной диаграмме (см. `beewings/core/indices.py`).

### Docker

```bash
docker build -t beewings-api .      # CPU-образ, веса копируются из runs/
docker run -p 8000:8000 beewings-api
# или: docker compose up --build   (volume ./data_io → /data, BEEWINGS_DATA_ROOT=/data)
```

`checkpoints/*.pt` в репозитории — это симлинки на `runs/*/best.pt`; Dockerfile
копирует реальные файлы весов напрямую.

---

## Тренировка с нуля

```bash
# 1. Собрать единый CSV (Y-координаты автоматически инвертируются под image origin)
beewings-ml-prepare \
    --root realData \
    --out data/wings19.csv \
    --n-points 19 \
    --relative-to realData

# 2. Тренировка (CUDA, ~2.5 часа на RTX 5080 для 100 эпох)
beewings-ml-train \
    --csv data/wings19.csv \
    --image-root realData \
    --out runs/unet19_v1 \
    --epochs 100 \
    --batch 24 \
    --base-channels 48 \
    --lr 1.5e-3

# 3. Eval на validation
beewings-ml-eval \
    --csv data/wings19.csv \
    --checkpoint runs/unet19_v1/best.pt \
    --image-root realData
```

---

## GUI хоткеи

| Клавиша | Действие |
|---|---|
| Левый клик | Поставить текущую точку |
| Перетаскивание | Сдвинуть точку |
| Правый клик на точке | Контекстное меню (удалить, неуверенно, пропустить) |
| `1`–`9`, `0` | Выбрать точку № N (`Shift+` для 10–19) |
| `←` / `→` | Предыдущая / следующая точка |
| `↑` / `↓` | Предыдущее / следующее изображение |
| `Backspace` / `Del` | Удалить текущую точку |
| `U` | Пометить точку как uncertain |
| `S` | Пропустить точку (skipped) |
| `Alt + ←/→/↑/↓` | Сдвиг текущей точки на 1 пиксель |
| `Alt + Shift + arrow` | Sub-pixel сдвиг (0.1 px) |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo |
| Средняя кнопка мыши drag | Pan канваса (перчатка) |
| `Space` + ЛКМ | Pan (альтернатива) |
| Колесо | Прокрутка / Pan |
| `Ctrl`/`Cmd` + колесо | Zoom к курсору |
| `Cmd 0` / `Cmd 1` / `Z` | Fit / 100% / Zoom к текущей |
| `F` | Mirror flip |
| `R` | Reset view |
| `T` | Overlay предыдущего крыла |
| `Ctrl+\` / `Ctrl+]` / `F11` | Скрыть левую / правую / обе панели |

После **🧠 ML авто-определить**:
- Точки с уверенностью < 0.45 автоматически помечаются как `uncertain` (для ручной проверки)
- В статус-баре: `ML: 19/19 точек, средняя уверенность 0.78 — проверь точки [7, 14]`
- Низкоуверенные точки попадают в список для review

---

## Структура проекта

```
beewings/
├── annotator/            # PyQt6 GUI
│   ├── main_window.py    # Главное окно, ML/CV кнопки, индексы
│   ├── canvas.py         # Zoom/pan/crosshair/magnifier
│   ├── side_panel.py     # Профили, точки, метрики
│   ├── image_list.py     # Список крыльев
│   ├── magnifier.py      # Лупа под курсором
│   └── enhancements.py   # CLAHE/контраст/инверсия отображения
├── core/                 # Схемы данных, профили, экспорт, индексы
│   ├── profiles.py       # Алпатов 8/12, Тофильский 19
│   ├── indices.py        # CI, DsA, RI и др.
│   ├── metrics.py        # Live метрики для GUI
│   ├── text_overlay.py   # PIL с Unicode для cv2-оверлеев
│   ├── schema.py         # Pydantic Landmark/WingAnnotation
│   ├── io_csv.py / io_tps.py / io_coco.py  # Экспорт
├── detector/             # Классический CV-детектор (резерв)
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── skeleton.py
│   ├── shape_model.py
│   ├── matching.py
│   ├── canonicalize.py
│   └── evaluate.py
├── ml/                   # PyTorch ML pipeline
│   ├── prepare.py        # TPS → unified CSV
│   ├── dataset.py        # Dataset + albumentations
│   ├── model.py          # UNet
│   ├── heatmap.py        # Gaussian encoding + sub-pixel decoding
│   ├── train.py          # Training loop с AMP + cosine LR
│   ├── inference.py      # checkpoint → predictions (+ TTA + confidence)
│   ├── batch.py          # Pakage CLI
│   ├── evaluate.py       # Stats vs CSV ground truth
│   └── report.py         # Paper-ready figures
└── api/                  # FastAPI REST-обёртка над ядром
    ├── app.py            # Приложение, CORS, /health, /models
    ├── registry.py       # Ленивый кэш моделей (per-model lock)
    ├── io.py             # Ввод картинки (upload | путь) + защита путей
    ├── schemas.py        # Pydantic DTO запросов/ответов
    ├── run.py            # Точка входа beewings-api (uvicorn)
    └── routers/          # detect / segment / indices / export
```

## Структура данных

```
data/
├── wings19.csv               # 7391 крыло (Тофильский)
├── wings12.csv               # 6935 крыльев (Алпатов)
└── wings19_images/           # Унифицированный пул изображений
                              # (обе схемы расшарят файлы)

runs/                         # Чекпоинты + train.log + config.json
├── unet19_v1/best.pt
└── alpatov12_v1/best.pt

checkpoints/                  # Production-ready
├── tofilski19.pt             # → runs/unet19_v1/best.pt
└── alpatov12.pt              # → runs/alpatov12_v1/best.pt

reports/                      # beewings-ml-report выходы
├── tofilski19_test/
│   ├── summary.json
│   ├── eval_table.csv
│   ├── per_landmark.png
│   ├── error_hist.png
│   └── sample_overlays/
└── alpatov12_test/
```

## Формат checkpoint

```python
{
    "model": state_dict,        # веса UNet
    "config": {                 # параметры тренировки
        "n_points": 12 | 19,
        "input_h": 256, "input_w": 512,
        "heatmap_h": 64, "heatmap_w": 128,
        "base_channels": 48,
        ...
    },
    "epoch": 100,
    "val": { "mean": ..., "median": ..., "p90": ... },
}
```

Проверка совместимости перед загрузкой:
```python
ckpt = torch.load("checkpoints/alpatov12.pt", weights_only=False)
assert ckpt["config"]["n_points"] == 12  # Alpatov
```

GUI делает эту проверку автоматически и блокирует cross-profile загрузку.
