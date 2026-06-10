# Доработки: удаление из списка, ресайз этикетки, дерево в «Точках» — дизайн

**Дата:** 2026-06-10
**Статус:** утверждён, готов к плану

## 1. Задачи

1. Удаление крыла клавишей **Del/Backspace** из списка «Найденные крылья».
2. Полноценное редактирование **рамки этикетки** (выбор, ресайз, перемещение,
   удаление) — наравне с крыльями.
3. Режим «Точки»: левая панель — **дерево** «картинка (сплит) → крылья»; клик по
   сплиту раскрывает его крылья и открывает первое; клик по крылу открывает
   именно его; навигация непрерывная.

## 2. Удаление из списка объектов (Нарезка)

`crop_page.objects` становится подклассом `QListWidget` (`_ObjectList`) с сигналом
`deleteRequested = pyqtSignal()`, эмитируемым на `Key_Delete`/`Key_Backspace`.
`CropPage` подключает его к `_delete_object()` (та же логика, что кнопка). Фокус на
списке → нажатие удаляет выделенное крыло; список и канвас обновляются.

## 3. Редактирование этикетки (box_canvas)

Этикетка (красная рамка) становится выбираемой и редактируемой:
- состояние `_label_sel: bool`; при выборе этикетки снимается выбор крыла и
  наоборот; новый сигнал `labelSelected = pyqtSignal(bool)`.
- перетаскивание унифицируется: `_drag = (kind, idx, handle)`, где
  `kind ∈ {"wing","label"}` (для label `idx` игнорируется); ресайз/перемещение
  через существующую `geometry.resize_box`/`hit_test`.
- `mousePressEvent`: сначала hit-test крыльев; если промах — hit-test этикетки
  (`hit_test([self._label], …)`); попадание → выбрать этикетку и начать drag.
- выделенной этикетке рисуются те же 8 маркеров (`_draw_handles`).
- `keyPressEvent`: если выбрана этикетка — стрелки двигают её (Shift ×10),
  `Del/Backspace` — `clear_label()`.
- этикетка по-прежнему исключается из нарезки; сохранение в `project.json` через
  существующий `boxesChanged`→`_sync_boxes`.
- сохраняются текущие `begin_label_draw`/`set_label`/`clear_label`/`make_label`.

## 4. Дерево «картинка → крылья» (LandmarkTab)

Левая панель вкладки «Точки» — `QTreeWidget`:
- верхний уровень: сплиты (по `project.scans`), подпись = имя файла;
- дети сплита: файлы крыльев `crops_dir(scan)/<stem>_crop_*.jpg` (исключая
  `_label`/`_debug`), подпись = имя крыла + прогресс (✓/●/○ как в `ImageList`);
  заполняются лениво при раскрытии; если папка крыльев пуста/нет — один
  неактивный пункт «(не нарезано)».
- клик по **сплиту** (родитель): раскрыть + открыть первое крыло;
- клик по **крылу** (ребёнок): `annot.show_image(scan_crops_dir, crop_path)` —
  при необходимости `load_folder`, затем выбрать именно это крыло; дальше ↑/↓ в
  аннаторе листают крылья непрерывно.
- внутренний список аннотатора во вкладке **скрыт** (`set_browser_visible(False)`).

## 5. AnnotatorWidget API

- `set_browser_visible(visible: bool)` — показать/скрыть внутренний `image_list`
  (для встраивания во вкладку, где навигацию ведёт дерево).
- `show_image(folder: Path, path: Path)` — если папка не текущая, `load_folder`;
  затем выбрать элемент `image_list`, совпадающий с `path` (через новый
  `ImageList.select_path(path) -> bool`), что триггерит загрузку крыла.
- `ImageList.select_path(path)` — найти строку по UserRole-пути и выставить
  currentRow; вернуть True/False.

## 6. Модули

- `beewings/pipeline/box_canvas.py` — выбор/drag/resize/клавиши этикетки;
  `labelSelected`; унификация `_drag` кортежа.
- `beewings/pipeline/pages/crop_page.py` — `_ObjectList` с `deleteRequested`;
  реакция на `labelSelected` (подсветка статуса этикетки).
- `beewings/annotator/image_list.py` — `select_path`.
- `beewings/annotator/annotator_widget.py` — `set_browser_visible`, `show_image`.
- `beewings/app/project_window.py` — `LandmarkTab` на `QTreeWidget`.

## 7. Тесты

- `box_canvas` (headless): программно выбрать этикетку (`select_label()` тест-хук
  или через смоделированный hit) → `labelSelected` эмитится; `clear_label` по
  Del-пути; ресайз этикетки через `resize_box` (юнит уже покрыт в geometry).
- `image_list.select_path` — выбирает правильную строку, возвращает False для
  отсутствующего пути.
- `annotator_widget` (headless): `set_browser_visible(False)` скрывает список;
  `show_image` выбирает крыло (грузит папку + выставляет путь).
- `crop_page` (headless): `_ObjectList.deleteRequested` → удаляет крыло.
- `LandmarkTab` (headless): дерево строит N сплитов; раскрытие нарезанного скана
  даёт его крылья; выбор крыла вызывает `annot.show_image`.

## 8. Вне scope

- Переход стрелками с последнего крыла одного сплита на следующий сплит
  (навигация остаётся в пределах текущего сплита).
- Мультивыбор.
