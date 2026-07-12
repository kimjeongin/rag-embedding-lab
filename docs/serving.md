# 서빙 — fine-tuned 모델 + Qdrant

랩에서 학습·평가한 dense 모델을 실제 검색으로 서빙하는 경로입니다. 구성은 두 조각:

- **임베딩 추론**: sentence-transformers **인프로세스** (서빙 결정 사항 — Ollama는 이 경로에 없음).
  학습 산출물(`outputs/…`)을 변환 없이 그대로 로드합니다.
- **벡터 검색**: **Qdrant**. 어댑터는 qdrant-client 없이 httpx로 REST를 직접 호출합니다
  (`rag/vectorstore/qdrant.py`) — 모든 와이어 호출이 코드에 보이도록.

## 빠른 시작

```bash
make qdrant                              # 로컬 Qdrant (docker, :6333)
make index SERVE_MODEL=outputs/embedding-ft   # 코퍼스 색인 + alias 전환
uv run rag-search "vpn 안됨"              # CLI 스모크 테스트
make serve-ft                            # /api/search 가 색인을 서빙
```

```bash
curl -s localhost:8000/api/search -H 'content-type: application/json' \
  -d '{"query": "연차 신청 방법", "top_k": 5}'       # 응답에 embed_ms/search_ms 지연 분해 포함
curl -s localhost:8000/api/search/status   # alias → collection, dim/모델 일치, 컬렉션 목록
curl -s localhost:8000/api/index -H 'content-type: application/json' -d '{}'  # 백그라운드 재색인
curl -s localhost:8000/api/index/status    # 재색인 진행률
curl -s localhost:8000/api/index/alias -H 'content-type: application/json' \
  -d '{"collection": "docs__옛모델__1024d__지문"}'   # 즉시 롤백 (재임베딩 없음)
curl -s -X POST localhost:8000/api/index/prune       # 라이브 외 컬렉션 삭제 (재색인 중이면 409)
```

웹 UI의 **검색 탭**이 이 전부를 시각화합니다: 인덱스 상태(가드 포함) · 재색인(진행바) ·
컬렉션 인벤토리(라이브 전환 = 롤백 · 정리) · 실검색(top-k · 지연 분해).

## 모델 교체 = 재색인, 자동

서로 다른 모델의 벡터는 같은 공간이 아니므로 모델이 바뀌면 전체 재임베딩이 필수입니다.
이를 컬렉션 버저닝으로 자동화했습니다 (`rag/serving.py`):

```
{prefix}__{model-slug}__{dim}d__{corpus-fingerprint}     ← 버전 컬렉션 (모델·차원·코퍼스 내용 인코딩)
{prefix}-live                                            ← 검색이 바라보는 유일한 이름 (alias)
```

`rag-index`는 (모델, 차원, 코퍼스 내용)에서 컬렉션 이름을 **결정적으로** 유도합니다:

1. 같은 이름의 컬렉션이 이미 다 차 있으면 → 임베딩 생략, alias만 보장 (**멱등** — 재실행 무해)
2. 없으면 → 새 컬렉션 생성, 배치 임베딩+upsert
3. 마지막에 alias를 **원자적으로** 전환 (Qdrant의 delete+create 단일 액션) — 검색은 절반만
   색인된 인덱스를 절대 보지 않고, 전환은 무중단
4. 이전 컬렉션은 롤백용으로 남음 → **롤백 = alias만 되돌리기** (검색 탭의 "라이브 전환"
   버튼 또는 `POST /api/index/alias` — 재임베딩 없이 즉시). 새 인덱스 확인 후
   `rag-index --prune`(또는 검색 탭의 "정리")으로 사본을 삭제

멱등이므로 자동화에 그대로 걸 수 있고, 실제로 걸려 있습니다:

- **핸드오프 훅**: 모델 페이지에서 핸드오프하면(`POST /api/models/handoff`) 그 모델로
  백그라운드 재색인이 자동 시작됩니다 — 핸드오프가 곧 "이 모델이 라이브로 간다"는 결정이므로.
  (`reindex: false`로 끌 수 있음)
- **서버 재색인 잡**: `POST /api/index`(모델 선택, 409 = 이미 실행 중) +
  `GET /api/index/status`(진행률 폴링). 웹 UI **검색 탭**에서 버튼/진행바로 노출됩니다.
- CLI도 동일 flow를 씁니다: `uv run rag-index --model outputs/새모델`.

## 포맷 패리티 (가장 중요한 계약)

색인의 문서와 검색의 쿼리는 학습·평가와 **같은** `rag/core/formatting.py`를 통과합니다
(Embedder 어댑터 내부에서) — 쿼리는 `Instruct: {task}\nQuery: {q}`, 문서는
`{title}\n\n{content}`. 서빙 쪽에서 이 포맷이 어긋나면 fine-tune 이득이 사라집니다.
Ollama 등 다른 스택으로 옮길 때의 검증 절차는 [serving-parity.md](serving-parity.md) 참고.

## 안전 가드

- **차원 가드**: 검색 시 인덱스 dim ≠ 임베더 dim이면 503 + 재색인 안내 (Matryoshka
  truncate 포함 — `rag-index --truncate-dim N`으로 색인했다면 서버도 `EMBED_TRUNCATE_DIM=N`).
- **모델 가드**: 같은 dim의 다른 모델은 기술적으로 검색이 되지만 순위가 무의미합니다.
  컬렉션 이름에 모델 슬러그가 박혀 있어 서버가 자동 비교합니다 —
  `/api/search/status`의 `model_matches: false`면 검색 탭이 amber 경고를 띄우고,
  재색인 모델을 서버 임베더와 다르게 고르면 **색인을 시작하기 전에** 경고합니다.
- Qdrant 다운/색인 없음 → `VectorStoreError` → HTTP 503 (원인과 다음 행동이 담긴 메시지).
- **정리 가드**: 재색인이 도는 동안 `POST /api/index/prune`은 409 — 잡이 만들고 있는
  (아직 라이브가 아닌) 컬렉션을 지워버리는 사고 방지.

## 상태 한눈에 (헤더)

웹 UI 헤더가 서빙의 "현재 세계 상태"를 상시 표시합니다: 프로세스 임베더(`ST`/`Ollama` +
모델명), Qdrant 생존 점등, 재색인 중 pill. Ollama는 죽어 있을 때만 `Ollama ↓`로
나타납니다(데이터 탭의 LLM 합성이 사용) — 건강한 상태는 조용히, 예외만 시끄럽게.

## 환경 변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 주소 |
| `QDRANT_COLLECTION` | `docs` | 컬렉션 패밀리 접두어 (alias는 `{prefix}-live`) |
| `EMBEDDER` | `sentence-transformers` | 서빙 임베더 백엔드 (기본이 ST — Ollama는 서빙 경로에 없음) |
| `ST_MODEL` | `outputs/embedding-ft` | 서빙할 모델 경로 (핸드오프한 모델) |
