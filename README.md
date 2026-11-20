# SkyrimLLM — construcción de corpus

Pipeline reproducible que construye los cinco corpus funcionales de SkyrimLLM (DS-LLM + RAG para
NPC de Skyrim), con alcance inicial **Carrera Blanca (Whiterun)**. Documentación completa:
[docs/CORPUS.md](docs/CORPUS.md) · cifras: [docs/corpus_stats.md](docs/corpus_stats.md).

## Estructura

```
configs/            sources.yaml (fuentes y licencias) · scope_whiterun.yaml · c4_generation.yaml
curation/           npc_overlays · secrets · world_effects · player_templates · gold_examples
skyrimllm_data/     código del pipeline (schemas, uesp, wikitext, build_c1..c4, retrieval,
                    prompting, quality, generate_c4, build_eval, assemble_sft, report, pipeline)
data/raw/           caché UESP (no versionar)
data/corpus/        c1_world · c2_profiles · c3_quests · c4_dialogue · eval · sft · MANIFEST.json
tests/              tests de parsers, control de acceso y validador
```

## Uso

```powershell
python -m venv .venv
$env:WITHOUT_EXTENSION = "1"; .venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m skyrimllm_data.pipeline --fetch
```

Generación sintética (Corpus 4) con un profesor local en Ollama:

```powershell
ollama pull qwen2.5:7b-instruct
.venv\Scripts\python -m skyrimllm_data.generate_c4 --limit 50 --paraphrase
.venv\Scripts\python -m skyrimllm_data.generate_c4 --paraphrase
.venv\Scripts\python -m skyrimllm_data.pipeline --from assemble
```

Cualquier endpoint compatible con OpenAI sirve (`TEACHER_BASE_URL`, `TEACHER_MODEL`,
`TEACHER_API_KEY`). Para probar el pipeline sin modelo: `--backend mock`.

Diálogo exportado de tu copia del juego (opcional, no redistribuible):

```powershell
.venv\Scripts\python -m skyrimllm_data.import_external --format alpaca --input ruta\skyrim_alpaca_style_dataset.json
```

## Atribución

Contenido derivado de [UESP](https://en.uesp.net) bajo CC BY-SA 2.5. El diálogo de *The Elder
Scrolls V: Skyrim* es © Bethesda Softworks y sólo se usa con fines de investigación.
