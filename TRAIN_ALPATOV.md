# Тренировка модели Алпатова (12 точек)

Использовать когда вернёшься в локальную сеть и сможешь зайти на сервер `192.168.0.111`.

## 1. Залить данные на сервер

```bash
# с Mac, из корня проекта
rsync -az data/wings12.csv beewings-server:~/beewings/data/
rsync -az data/wings12_images/ beewings-server:~/beewings/data/wings12_images/

# проверить место
ssh beewings-server 'df -h ~ | tail -1'
```

## 2. Запустить тренировку

```bash
ssh beewings-server '
cd ~/beewings
nohup ~/miniconda3/envs/ai/bin/python -m beewings.ml.train \
    --csv data/wings12.csv \
    --image-root data/wings12_images \
    --out runs/alpatov12_v1 \
    --n-points 12 \
    --epochs 100 \
    --batch 24 \
    --base-channels 48 \
    --workers 6 \
    --lr 1.5e-3 \
    > runs/alpatov12_v1_console.log 2>&1 &
echo "pid: $!"
'
```

Время: ~2.5-3 часа на RTX 5080.

## 3. Скачать checkpoint

```bash
# на сервере: упаковать слим-архив (50 MB)
ssh beewings-server '
cd ~/beewings
tar -czvf ~/Desktop/alpatov12_trained_$(date +%Y%m%d_%H%M).tar.gz \
    runs/alpatov12_v1/best.pt \
    runs/alpatov12_v1/train.log \
    runs/alpatov12_v1/config.json
ls -lh ~/Desktop/alpatov12_trained_*.tar.gz
'
# скопировать на Mac (или вручную через флешку)
scp 'beewings-server:~/Desktop/alpatov12_trained_*.tar.gz' .
```

## 4. Распаковать и установить

```bash
# на Mac, в корне проекта
tar -xzvf alpatov12_trained_*.tar.gz
cp runs/alpatov12_v1/best.pt checkpoints/alpatov12.pt
ls -lh checkpoints/
# должно быть:
#   tofilski19.pt
#   alpatov12.pt
```

## 5. Использовать в GUI

Запустить `.venv/bin/beewings`, в правой панели в селекторе профиля выбрать **«Алпатов 12 точек»** и нажать **🧠 ML авто-определить точки**. Модель применит свои 12 точек.

При выборе **«Алпатов 8 точек»** будет использована та же модель Алпатова, но GUI покажет только первые 8 точек (они образуют исходную классическую схему Алпатова).

## Почему отдельная модель?

Алпатов и Тофильский — **разные методики**:
- У них разные биологические опорные точки.
- Их нумерация **независима** — Алпатов L1 ≠ Тофильский L1.
- Индексы и классификаторы пород ожидают **строго свою нумерацию**.

Использовать модель Тофильского для Алпатовских вычислений = тихо испортить определение породы. Поэтому GUI блокирует cross-загрузку checkpoint'ов с проверкой `n_points`.
