# SkyrimLLM — Corpus de entrenamiento (datasheet)

Este documento describe **qué contiene cada corpus, de dónde sale, cómo se construye y cómo se
valida**. Las cifras actualizadas están en [corpus_stats.md](corpus_stats.md) (se regeneran con
`python -m skyrimllm_data.report`) y los hashes de cada fichero en `data/corpus/MANIFEST.json`.

## 1. Principio de diseño

El DS-LLM se entrena para **comportarse como NPC usando el contexto**, no para memorizar el lore.
Por eso todos los ejemplos SFT (Corpus 4) se construyen con *exactamente* el mismo prompt que verá
el sistema en ejecución (`skyrimllm_data/prompting.py`, versión `npc-ctx-v1`):

```
perfil NPC (C2) + estado del mundo (C3) + hechos recuperados y filtrados (C1) + recuerdos (C5)
        └──────────────────────────── system prompt ───────────────────────────┘
jugador → user        ·        respuesta del NPC → assistant
```

Así el modelo aprende a: usar los fragmentos recuperados, ignorar los irrelevantes, admitir
desconocimiento, guardar secretos y reaccionar al estado de la partida.

## 2. Los cinco corpus

| Corpus | Fichero(s) | Destino | Fuente |
|---|---|---|---|
| C1 World Knowledge | `c1_world/{documents,chunks,entities,relations}.jsonl` | Knowledge RAG | UESP (espacios *Skyrim* y *Lore*), API en vivo con `revision_id` |
| C2 NPC Profiles | `c2_profiles/npcs.jsonl` | NPC Profile System | UESP `{{NPC Summary}}` + lead + diálogo citado · bios Mantella (secundarias) · `curation/npc_overlays.yaml` |
| C3 Quests & World State | `c3_quests/{quests.jsonl, quest_index.json, world_state_schema.json}` | World State Manager | UESP `{{Quest Header}}` + `{{Journal Entries}}` · `curation/world_effects.yaml` |
| C4 Dialogue & Behavior | `c4_dialogue/{scenarios,seed_uesp_dialogue}.jsonl`, `generated/`, `external/` → `sft/{train,val}.jsonl` | SFT del DS-LLM | escenarios deterministas + profesor LLM local + diálogo real citado en UESP + (opcional) export propio del juego |
| C5 NPC Memory & Events | sólo esquema `MemoryEvent` (`schemas.py`) | NPC Memory System | se genera en ejecución; en SFT se inyectan recuerdos sintéticos |
| Evaluación | `eval/eval_cases.jsonl` | Evaluación | escenarios retenidos + sondas por dimensión |

Todos los registros llevan `provenance` (fuente, URL, revisión, licencia, `redistributable`,
transformación aplicada). El registro de fuentes y las decisiones de inclusión/exclusión está en
[`configs/sources.yaml`](../configs/sources.yaml).

## 3. Control de conocimiento (clave para "restricciones de conocimiento")

Cada fragmento de C1 tiene **ámbitos** (`scopes`) y, si procede, **compuertas de misión**
(`quest_gates`). Cada NPC de C2 tiene sus ámbitos. Un NPC sólo puede recuperar un fragmento si
comparten algún ámbito y todas las compuertas se cumplen en el `WorldState` actual
(`retrieval.visible`).

| Ámbito | Quién lo conoce | Asignación |
|---|---|---|
| `common` | cualquier habitante de Skyrim | leads de ciudades/regiones mayores, dioses, razas, provincias, facciones famosas (`COMMON_LORE`), criaturas |
| `regional:<comarca>` | habitantes de la comarca | lugares menores según `region` de la infobox |
| `local:<asentamiento>` | vecinos | NPC de ese asentamiento (lead) |
| `faction:<id>` | miembros | facciones legibles del NPC + overlays |
| `scholarly` | magos, sacerdotes, bardos, corte… | libros, historia profunda del espacio *Lore* |
| `personal:<npc>` | sólo ese NPC | secciones narrativas de su página, secretos |
| `quest_participants` + compuerta | NPC implicados, tras iniciar la misión | walkthroughs (spoilers) |
| `meta` | nadie (se descarta) | mecánicas, objetos con estadísticas |

Además: secciones *meta* (bugs, notas, estadísticas, mercancía…) se eliminan, frases de mecánica
de juego se filtran (`META_SENT`), y `curation/secrets.yaml` **estrecha** el ámbito de cualquier
fragmento que filtre un secreto (p. ej. que Saadia es *Iman*, o que el Círculo de los Compañeros
son licántropos). El informe auditable está en `c1_world/restriction_report.json`.

## 4. Corpus 4: tipos de comportamiento

16 comportamientos × NPC × preset de partida (`configs/c4_generation.yaml`):
`smalltalk, self_identity, occupation_service, local_knowledge, world_lore, opinion, unknown_info,
restricted_info, quest_aware, world_state_reaction, memory_recall (con y sin recuerdos),
multi_turn, out_of_world, hostile_player, player_race_reaction, rumor`.

Presets de partida: `arrival, dragonborn, companion, stormcloak_whiterun, imperial_whiterun,
late_game`; cada uno fija misiones/etapas y deriva variables del mundo (quién gobierna, títulos
del jugador…). Raza/género/nombre del jugador se muestrean.

Robustez RAG: con probabilidad 0,5 se inserta un fragmento **distractor** visible pero irrelevante
(estilo RAFT) y en `smalltalk/out_of_world/hostile_player` no se aporta contexto.

### Fuentes de ejemplos

1. **Diálogo real citado en UESP** (`seed_uesp_dialogue.jsonl`): cada línea se contextualiza; la
   condición del wiki ("cuando Carrera Blanca está bajo control Capa de la Tormenta", "si eres
   Dunmer…") se traduce a preset/raza. Anclas de estilo auténticas, **no redistribuibles**.
2. **Sintético** (`generate_c4.py`): profesor LLM vía API compatible OpenAI (Ollama por defecto)
   con *few-shot* de `curation/gold_examples.yaml`, nota de dirección por comportamiento, y
   validación en dos pasos (reglas + juez LLM).
3. **Export propio del juego** (`import_external.py`): CSV de xEdit/Lazy Voice Finder o el JSON
   Alpaca de Mantella-LLM-Fine-Tuning; limitado al 25 % de la mezcla para que el modelo no aprenda
   a ignorar el contexto.

### Validación automática (`quality.py`)

Rechazo duro: respuesta vacía/larga, referencias meta (IA, NPC, juego, notas `[n]`), prefijo de
hablante, **fuga de secretos**, falta de incertidumbre cuando el NPC no sabe/no recuerda, ≥2
nombres propios no presentes en contexto ni en C1 (alucinación). Avisos: anacronismos, narración
larga, posibles fugas. Deduplicación por Jaccard de trigramas por NPC. Juez LLM (persona,
grounding, reglas, naturalidad ≥ 4).

## 5. Particiones y evaluación

- **NPC retenidos** (`eval_holdout_npcs`): nunca entran en train/val → mide generalización a NPC
  nuevos de Carrera Blanca.
- 5 % de escenarios de NPC vistos → `eval`; 5 % → `val`.
- `eval_cases.jsonl` añade sondas comprobables: preguntas factuales con anclas verificadas en UESP,
  sondas de fuga de secretos a NPC no autorizados, pares de contraste de estado del mundo (Imperial
  vs Capa de la Tormenta), memoria con/sin recuerdo, inyección de prompt y anacronismos, y líneas
  reales del juego como respuesta de referencia. Cada caso guarda el `system_prompt_snapshot`.

## 6. Revisión humana

- `review_status` en perfiles: `auto` → `curated` → `reviewed` (tras cotejar con la página UESP).
- Muestreo recomendado: 5 % de `generated/train.jsonl` estratificado por comportamiento, más todo
  lo marcado con avisos (`quality.flags`) en `restricted_info` y `unknown_info`.
- Los 27 overlays curados están redactados a partir de UESP y requieren una segunda revisión.

## 7. Licencias

- UESP: **CC BY-SA 2.5** → los corpus derivados (C1–C3, sintéticos) deben publicarse con CC BY-SA
  y atribución a UESP.
- Bios Mantella: AGPL-3.0 (campo separado `bio_secondary`).
- Diálogo de Bethesda (citado en UESP o exportado del juego): uso de investigación, marcado
  `redistributable=false`; `assemble_sft --public` produce una versión sin él.

## 8. Reproducir

```
python -m skyrimllm_data.pipeline --fetch           # descarga UESP + todos los corpus
python -m skyrimllm_data.generate_c4 --limit 50     # piloto con el profesor
python -m skyrimllm_data.generate_c4                # generación completa (reanudable)
python -m skyrimllm_data.pipeline --from assemble   # mezcla final + estadísticas
```
