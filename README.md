# Native RAG для документации

Чистая реализация Native RAG для обычного текста и документации. Проект не обучает модель и не содержит специальной логики для исходного кода.

## Что используется

- локальная Transformers causal/chat-модель через `GeneratorBackend`;
- чанки Markdown/TXT с overlap и сохранением источника/заголовка;
- lexical BM25-поиск;
- optional native dense-представления из скрытого состояния той же загруженной модели;
- Reciprocal Rank Fusion (RRF) для объединения lexical и native-dense сигналов;
- capability-based chunked prefill и model-native KV-cache;
- compatibility fallback через обычный `generate()` для backend’ов без native execution;
- 4-bit NF4-квантование весов с FP16 compute по умолчанию, чтобы полный длинный контекст помещался в 12 ГБ VRAM;
- `fp16` как потенциально более быстрый, но менее экономный по весам режим;
- `fp32` как baseline из-за обнаруженной нестабильности Qwen3.5 GDN в `bf16` на текущем Windows/CUDA-стеке;
- reasoning переключается флагом `--thinking`.

Qwen3.5 остаётся reference backend’ом текущего проекта. Для другой стандартной Transformers-модели передайте её каталог через `--model`; выбор backend’а выполняется через `--backend auto|generic|qwen35`.

## Быстрый запуск

Из каталога проекта:

```powershell
python -m pip install -e .
python -m native_rag.cli index --docs .\docs --out .\indexes\docs
python -m native_rag.cli ask --index .\indexes\docs --question "Что описано в документации?"
```

По умолчанию индекс строится в portable BM25-only режиме и не загружает модель. Если нужен native dense-сигнал от той же LLM, явно укажите `--native-dense` при индексации и используйте совместимый backend при `ask`:

```powershell
python -m native_rag.cli index --docs .\docs --out .\indexes\docs --native-dense --model ..\Qwen3.5-0.8b
```

Для reasoning:

```powershell
python -m native_rag.cli ask --index .\indexes\docs --question "Сравни два подхода" --thinking
```

Для диагностики `bf16`:

```powershell
python -m native_rag.cli smoke --dtype bf16
```

Проверка квантованного режима:

```powershell
python -m native_rag.cli smoke --dtype nf4
```

Сравнение скорости без 4-bit-квантования:

```powershell
python -m native_rag.cli smoke --dtype fp16
```

FP32 baseline:

```powershell
python -m native_rag.cli smoke --dtype fp32
```

## Архитектура

```text
documents/*.md, *.txt
        │
        ▼
DocumentLoader → structural chunks → persisted Index
                                      ├─ BM25
                                      └─ optional native features from GeneratorBackend
        │
        ▼
HybridRetriever (RRF)
        │
        ▼
Neighbor expansion → PromptSpec → GeneratorBackend
                                  ├─ model tokenizer
                                  ├─ optional chunked prefill/cache
                                  └─ compatibility generate()
                                      ↓
                                    answer
```

Чанки добавляются в контекст перед вопросом, затем один раз префиллятся и переиспользуются через KV-cache. В retrieval нет понятия «текущего будущего чанка»: запрос и корпус разделены на уровне документов.

После retrieval по умолчанию добавляется один соседний чанк из того же файла (`--neighbor-radius 1`). Это восстанавливает локальный контекст на границах чанков; соседние файлы не смешиваются. `--candidate-k` задаёт более глубокий пул кандидатов, а `--max-context-chunks` — итоговый бюджет: сначала сохраняется широкий слой лучших якорей, затем небольшой резерв заполняется ближайшими соседями, после чего фрагменты возвращаются в порядке исходного документа.

Для документов с важными ответами на границах чанков есть двухуровневый режим: поиск остаётся на мелких чанках, а перед моделью только реально выбранные соседние чанки объединяются в spans через `--span-size`. Например, при индексе с чанками около 128 токенов `--span-size 2` старается передавать последовательные блоки примерно по 256 токенов, но одиночные далёкие anchors не теряются. Бюджет остаётся ограничен количеством мелких чанков, поэтому число итоговых spans может быть больше расчётного числа крупных блоков.

Для ручной калибровки можно, например, использовать `--candidate-k 64 --max-context-chunks 32 --neighbor-radius 1`: retrieval смотрит глубже 32 чанков, но в модель передаётся только ограниченный активный контекст.

Практический вариант для проверки двухуровневой схемы: индексировать документацию более мелкими чанками, затем задать `--span-size 2 --candidate-k 128 --max-context-chunks 32 --neighbor-radius 1`. В этом случае поиск смотрит 128 мелких кандидатов, а активное окно модели ограничено эквивалентом 64 мелких чанков; соседние выбранные фрагменты склеиваются адаптивно.

## Почему `bf16` не включён по умолчанию

В старом экспериментальном каталоге `repro_q35.py` обычный forward Qwen3.5 без RAG падал в `torch_chunk_gated_delta_rule` с `CUDA illegal memory access`. Причиной оказался in-place update в torch fallback GDN. В проекте он заменён на функционально эквивалентное обновление без записи в CUDA view; математика проверена сравнением с upstream fallback на CPU.

Для `bf16` дополнительно выполняется нормализация Q/K в FP32 перед FP32 accumulator path. На текущей машине после этого патча 1M corpus + 32k active window прошёл с правильным ответом, но занял около 82 секунд и достиг 11.58 GB reserved VRAM. Поэтому `bf16` пока остаётся экспериментальным режимом; рабочим default остаётся NF4.

Рабочий режим для RTX 3060 — `nf4`: bitsandbytes квантует веса при загрузке в NF4, а вычисления идут в FP16. Файлы исходной BF16-модели не изменяются. Это уменьшает память весов, но не отменяет стоимость глобального KV-cache на длинном контексте. Если приоритет — скорость и активное окно помещается, стоит сравнивать его с `fp16`: 4-bit не гарантирует более быстрый inference.

## Почему не `FP8`

На текущей RTX 3060 (compute capability `8.6`) нет аппаратного FP8 Tensor Core пути. Принудительно хранить отдельные тензоры в `float8` не превращает этот inference в полноценный FP8-режим и не исправляет сбой в GDN. Поэтому в проекте оставлены `nf4` как рабочий режим, `fp32` как baseline и экспериментальный `bf16`; FP8 имеет смысл тестировать на GPU с соответствующей аппаратной поддержкой и отдельным стеком квантования.

## Проверки

```powershell
python -m pytest
```

Тесты покрывают разбиение документов, overlap, BM25/RRF и сериализацию индекса. GPU smoke-тесты намеренно не делают скриншоты.

Прямой тест длинного контекста без RAG:

```powershell
python benchmarks\needle.py --length 260000 --dtype fp32
```

Он не строит индекс и не выполняет retrieval: одна контрольная строка помещается в синтетический контекст, после чего модель должна вернуть её шестизначный код. По умолчанию benchmark использует `nf4`; для FP32 baseline передайте `--dtype fp32`. В выводе сохраняются фактическая длина prompt, время, пик VRAM и `needle_hit`.

Проверка Native RAG с корпусом 250k и ограниченным активным окном:

```powershell
python benchmarks\active_window.py --dtype nf4 --corpus-tokens 250000 --budgets 2048 4096 16384 32768
```

Корпус остаётся вне prompt и KV-cache. В модель попадают только выбранные 256-токенные чанки: 8, 16, 64 или 128 соответственно. Retrieval в этом изолированном тесте — CPU-only lexical gate; дополнительных нейросетей и dense-проходов нет.

Синтетический multi-needle тест retrieval на корпусе в 1M токенов:

```powershell
python benchmarks\multi_needle_1m.py --corpus-tokens 1000000 --samples 3 --needles 8 --budget-tokens 8192 --chunk-tokens 128 --span-size 2 --candidate-k 128 --neighbor-radius 1
```

Этот тест не загружает веса модели: локальный tokenizer нужен только для точного размера корпуса в токенах. В разные зоны корпуса помещаются восемь документационных фактов, рядом с ними — похожие draft-фрагменты. Benchmark отдельно показывает seed-recall, попадание в активное окно, полное покрытие каждой иголки, худший ранг, позиционную recall и точность выбранных чанков. JSON сохраняется в `runs\multi_needle_1m_retrieval.json`. Позже тот же сценарий можно использовать для генерационного теста 1.5B.

Генерационный multi-needle прогон на локальной модели:

```powershell
python benchmarks\multi_needle_model.py --model ..\Qwen2.5-Coder --dtype nf4 --samples 3 --corpus-tokens 1000000 --budget-tokens 8192 --chunk-tokens 128 --span-size 2 --candidate-k 128 --neighbor-radius 1 --max-new-tokens 128
```

В отчёте отдельно сохраняются recall retrieval и recall ответов модели. Поэтому модель, которая получила все иголки, но не смогла вернуть их коды, не маскирует проблему поиска.

Проверка 32k окна на корпусе 1M с иголкой в центре окна:

```powershell
python benchmarks\active_window.py --dtype nf4 --corpus-tokens 1000000 --budgets 32768 --selection needle-centered
```

На текущем Windows/CUDA fallback этот 32k-прогон может завершиться `CUDA illegal memory access` внутри GDN kernel. Это нестабильность backend/kernel, а не переполнение активного окна и не размер внешнего корпуса.
