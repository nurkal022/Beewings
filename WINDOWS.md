# Запуск BeeWings на Windows

## 1. Установить Python (один раз)
Скачай **Python 3.10–3.13** с https://www.python.org/downloads/
При установке обязательно поставь галку **«Add Python to PATH»**.

Проверь в PowerShell / cmd:
```
python --version
```

## 2. Получить проект
```
git clone https://github.com/nurkal022/Beewings.git
cd Beewings
```
(или скачай ZIP с GitHub и распакуй)

## 3. Автоматическая настройка
Двойной клик по **`setup_windows.bat`**.

Скрипт сам:
- создаст виртуальное окружение `.venv`;
- установит все зависимости (PyTorch, PyQt6 и т.д. — ~1 ГБ, нужен интернет);
- скачает ML-модели в `checkpoints\` из GitHub Releases.

## 4. Запуск
Двойной клик по **`run_windows.bat`** — откроется приложение.

---

### GPU (необязательно)
По умолчанию ставится CPU-версия PyTorch — работает везде. Если на ПК есть видеокарта NVIDIA и нужно ускорение:
```
.venv\Scripts\activate
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Если модели не скачались
Скрипт тянет их отсюда:
https://github.com/nurkal022/Beewings/releases/tag/models-v1
Скачай `alpatov12.pt` и `tofilski19.pt` вручную и положи в папку `checkpoints\`.
