# 실데이터 이어받기 런북 (회사 적용)

> **이 문서 하나로** 새 세션 — 사람이든 새 Claude 세션이든 — 이 랩을 **실데이터로 이어받을 수 있게** 만든
> 자기완결 런북이다. 이전 대화 맥락도, 개인 메모리도 없이 **레포만 보고 시작**하는 것을 전제로 하므로,
> 필요한 "왜"까지 이 안에 담았다. 리허설(합성/대조군 데이터)은 끝났고, 남은 것은 실데이터를 붓는 것뿐이다.

---

## 0. START HERE — 지금 어디에 있나

랩은 **데이터 생성 → 학습 → 평가(통계검정) → 비교 → 납품 → 서빙**의 전 구간과, 그 결과를 믿게 하는
검증 장치(포맷 패리티·dev/final 분리·eval 지문·paired 검정·서빙 벤치)를 **전부 구축·리허설 완료**했다.
공개 프록시 데이터(korea.kr PoC)와 가상 인트라넷 대조군으로 파이프라인·측정기·절차를 검증했다.

**로드맵 상 현재 위치:** `Step 0(리허설) 완료` → **`Step 1(실데이터 연결)` 시작 지점.**

> 한 줄 요약: **새로 발명할 것은 없다. 실데이터가 연결되는 순간 준비된 코드로 며칠 안에 따라붙는다.**
> 병목은 "능력"이 아니라 "데이터"이고, 데이터는 조직 승인(§3 Step 1)이 리드타임을 지배한다.

---

## 1. 이 프로젝트가 무엇이고 왜 하는가 (메모리 대체 — 반드시 먼저 읽을 것)

- **성격:** 상사 보고용 과제. 평가는 결과도 보지만 **과정을 가장 중요하게** 본다.
- **두 주력 목표:**
  1. 임베딩 **파인튜닝(FT)이 실제로 사내검색 품질을 올렸는가.**
  2. 새 공개 임베딩 모델과 비교해 **차기 모델로 무엇을 선택할 것인가** (정확도 우선 — 비용 차이는 이 서비스 환경에선 무의미).
- **보조 목표:** 하이브리드 서치(BM25+dense) 성능 향상.
- **랩의 역할 = dense 부품만 납품.** 프로덕션은 이미 `BM25 + dense + 리랭커` 3단 하이브리드가 돈다.
  랩의 Qdrant 서빙 경로는 **레퍼런스/절차서**이지 배포 대상이 아니다. "적용 = 검증된 dense 모델 + 포맷 계약을
  기존 하이브리드에 부품 교체."
- **범위 밖(하지 말 것):** 에이전트 워크플로우, 문서 청킹(사내 문서가 짧고 페이지 단위로 자족적 — 청킹은
  오히려 문맥을 쪼개 손해). 이 둘은 재론하지 않는다.
- **주 지표 = recall@K.** dense의 임무는 "정답을 1등으로"가 아니라 **"정답을 상위 후보(리랭커에 넘길 K개)에
  넣기"**. 후보에 못 든 정답은 리랭커도 못 본다 → recall@K가 검색 품질의 천장.

---

## 2. 절대 깨면 안 되는 불변 (깨지면 **조용히** 망가진다 — 예외가 안 난다)

1. **포맷 패리티 / ModelProfile** (`rag/core/formatting.py`, `rag/modelprofile.py`): 학습·평가·서빙이
   **같은 포맷 코드**를 통과해야 한다. 틀리면 예외 없이 점수만 나빠진다 — 실측으로 **틀린 포맷의 FT 모델이
   학습조차 안 한 base보다 못했다**(0.345 vs 0.543). 새 모델을 붙이면 프로파일부터 확인.
2. **dev/final 규율:** dev로 모델을 **선택**하고, final은 **승자 1회만** 확정한다. **final로 튜닝하는 순간
   선택 편향으로 숫자가 오염**된다. (실측 교훈: dev에서 보인 하이브리드 이득이 final 천장에선 비재현 —
   이 규율이 그걸 정직하게 잡아냈다.)
3. **eval 지문(fingerprint):** 내용이 다른 평가셋끼리는 비교가 **자동 차단**된다. 평가셋을 바꾸면 과거
   점수와 직접 비교하지 말 것.
4. **판정 규율:** recall@50이 1.0으로 **포화하면 그 지표로 모델을 가릴 수 없다** → `nDCG@10`·`recall@5`로
   판정. **paired 순열 검정(p값) 없는 Δ는 주장하지 말 것** (`rag.diff` / 실험 탭).
5. **재현성:** 모든 학습은 레시피·데이터 지문·epoch별 점수를 `train_meta.json`에 자동 기록한다. 데이터를
   재생성하면 지문이 바뀐다(정상).

---

## 3. 임계경로 — 순서대로 (순서가 곧 의존성)

### Step 1 — 실데이터 연결  ★제1 블로커
**선행(지금 당장):** **PII/정보보안 사인오프.** "검색 로그를 학습에 쓴다"는 승인은 코드가 아니라 조직
절차이고 리드타임이 가장 길다. **가장 먼저 요청을 걸어라** — 전체 일정이 여기서 결정된다.

**데이터 디렉토리 관례:** 실데이터는 새 디렉토리(예: `data-company/`)에 담고 env로 스위치한다.
`data/`(PoC)·`data-intranet/`(대조군)은 그대로 둔다(참조용).

| 대상 | env | 기본값 |
|---|---|---|
| 학습쌍 (train) | `TRAIN_FILE` | `data/train.jsonl` |
| 학습쌍 (검증) | `TRAIN_EVAL_FILE` | `data/test.jsonl` |
| 평가셋 (BEIR) | `EVAL_DIR` | (env로 지정) |

**a) corpus 반입:** 사내 사이트 크롤(`make crawl CRAWL_URL=…`) 또는 사내 덤프를 `corpus.jsonl`로.
payload 11필드(site_id·url·version_name·title·title_eng·llm_title·description·description_eng·
user_queries·need_steps·hard_guide_name)는 **이미 실 스키마와 sync됨**(`rag/datagen/intranet.py` 참조).

**b) 클릭로그 반입 & 클리닝:** 실로그는 그대로 쓰면 안 된다(포지션 바이어스·오클릭·재검색·PII).
클리닝 계층은 **이미 만들어 리허설 완료** — 실로그와 모의로그가 **같은 코드**를 통과한다.
- **UI 경로:** 데이터 탭 → "실데이터 가져오기" → **클릭로그(세션) 모드**에 원본 붙여넣기 → 규칙별 처리
  내역(직접/전이/바운스/PII) 확인 후 학습쌍으로 변환.
- **코드 경로:** `rag.datagen.clicklog.clean(events)` — dwell 필터·**재검색 전이**(은어 supervision의 전부:
  실패한 앞 쿼리를 최종 만족 문서에 연결)·skip-above hard negative 채굴·(쿼리,문서) 집계·PII 차단.
- **이벤트 포맷 참고:** `data-intranet/clicklog.jsonl` (세션 이벤트 JSONL).

**완료 기준(DoD):** `data-company/corpus.jsonl` 존재 + 클리닝된 train 쌍 생성 + PII 차단 동작 확인.

---

### Step 2 — 실 평가셋 구축
**⚠️ 신규 구현 항목 — 시간분리(temporal split).** 현재 코드의 dev/final 분리는 문서/쿼리 단위다.
실로그는 **과거→학습 / 미래→평가**로 **시간 컷**을 해야 누수가 없고 실배포를 흉내 낸다.
- 착수 지점: `rag/datagen/eval_from_corpus.py`·`eval_corpus.py` 근처에서 qrels `dev.tsv`/`final.tsv`를
  타임스탬프 기준으로 나누는 로직 추가.

**슬라이스 태그:** `queries.jsonl`의 `slice` 필드에 쿼리 **출처**(검색로그 / 신규라벨링 / 재검색회수)를
달아라 — 은어 슬라이스 가시성이 여기서 나온다(전체 평균은 이봉 분포를 가린다).

**포화 방지:** 평가 corpus는 **사이트 전체를 haystack**으로(`make evalset`), 규모는 **수천 단위**.
`recall@50`이 1.0이면 corpus를 더 키워라(실측: 300→1500에서 비로소 지표가 모델을 가리기 시작).

**완료 기준(DoD):** `EVAL_DIR`에 시간분리된 dev/final qrels + 슬라이스 태그, `recall@50` 비포화.

---

### Step 3 — 실데이터 학습 + 모델 결정
- **hard negative 마이닝 활성화:** 클리닝 계층이 skip-above로 hard negative를 부산물로 채굴한다.
  `TRAIN_MAX_NEGATIVES`로 컬럼 상한을 두고(메모리 주의: negative 1컬럼 = 배치 텍스트 배증), Triplet/GIST와 결합.
- **CUDA 튜닝 스윕:** **[docs/model-tuning-plan.md](model-tuning-plan.md) 그대로 실행**(LR×batch → loss →
  dropout → Matryoshka → hard negative → LoRA, median pruning, dev/final + paired 검정). MPS OOM 시
  `TRAIN_GRAD_CHECKPOINT=1`(레시피 보존 — 결과 불변).
- **base vs FT 비교:** `ST_MODEL`을 스왑해 `rag-eval` 두 번(같은 corpus·지표) → paired 검정.
- **공개 모델 서베이 재확인:** `scripts/model_survey.py` — bge-m3·e5·KURE 등 공개 모델 zero-shot이
  **실데이터에서도 은어 슬라이스에서 참패하는지**(리허설 결론: 한국어 특화 KURE조차 은어 0.19). 이게
  "차기 모델 = 무엇을 파인튜닝할까"의 핵심 근거.
- **하이브리드 재측정:** `scripts/hybrid_sweep.py` — FT 위에 RRF 융합 이득. ⚠️ 랩 BM25는 문자 bigram
  대역이므로, **진짜 개선 폭은 프로덕션 형태소 BM25로 바꿔 재측정**해야 방향이 맞는다.
- **final 확정:** dev 승자를 **final 1회**만.

**완료 기준(DoD):** 승자 모델 + `train_meta.json`, base 대비 paired **p값**, 서베이·하이브리드 표 갱신.

---

### Step 4–6 — 출시·검증·운영 (Step 3과 병렬 준비)
보고서 핵심(두 목표)은 Step 1–3에서 답이 나온다. 아래는 **실제 배포**의 임계경로 — 3과 병렬로 준비만 걸어둔다.

- **Step 4 · 서빙 통합 + ACL:** dense 부품을 **프로덕션 하이브리드에 교체**(랩 Qdrant 아님). 납품물 =
  가중치 + **포맷 계약(ModelProfile)** + 패리티 벡터([docs/serving-parity.md](serving-parity.md)).
  **ACL/권한 필터**(Qdrant payload `site_id`·권한)는 출시 게이트 — 볼 수 없는 문서를 노출하면 안 된다.
- **Step 5 · 리랭커 공동검증 + A/B:** dense만 바꾸면 리랭커가 이미 상쇄 중일 수도, 재학습이 필요할 수도.
  **실 리랭커를 루프에 넣고 BM25 상보성을 end-to-end로 재측정** + **A/B로 서비스 지표**(CTR·재검색률·
  무결과율·dwell). 랩 점수는 필요조건이지 충분조건이 아니다.
- **Step 6 · 운영:** 드리프트 모니터링(무결과율·재검색률 급증 = 신조어 신호 → 재학습 트리거) + 롤백 런북
  (alias 즉시 롤백은 구현됨 — "언제 롤백하나"의 임계 기준만 문서화).

---

## 4. 명령 빠른참조

```bash
# 학습 (실데이터 스위치 + hard negative + 프로파일 자동해석)
TRAIN_FILE=data-company/train.jsonl TRAIN_EVAL_FILE=data-company/test.jsonl \
TRAIN_BASE_MODEL=Qwen/Qwen3-Embedding-0.6B TRAIN_MAX_NEGATIVES=1 uv run rag-train

# 평가 (base vs FT — ST_MODEL만 바꿔 두 번, 같은 EVAL_DIR)
EVAL_DIR=data-company/eval ST_MODEL=Qwen/Qwen3-Embedding-0.6B uv run rag-eval   # base
EVAL_DIR=data-company/eval ST_MODEL=outputs/embedding-ft       uv run rag-eval  # FT

# 공개 모델 서베이 / 하이브리드 융합 스윕 (레지스트리 자동 기록)
uv run python scripts/model_survey.py
uv run python scripts/hybrid_sweep.py

# 서빙 리허설 (레퍼런스 절차 — 프로덕션 교체 전 검증용)
make qdrant && make index SERVE_MODEL=outputs/embedding-ft && make serve-ft
uv run rag-bench --model outputs/embedding-ft --label company-ft   # 지연·GPU·저장 벤치
```

---

## 5. 산출물 갱신 (보고 — 계속 업데이트 대상)
실데이터 결과가 나오면 **1급 산출물**을 갱신한다:
- **[docs/report.md](report.md)** — §3에 실데이터 결과 절 추가, §4 로드맵 갱신.
- **보고 페이지**(`frontend/src/routes/Report.tsx`) — 헤드라인·표·타임라인·향후계획.
- **About 페이지**(`frontend/src/routes/About.tsx`) — 개념 참고(공부용).

---

## 참조 맵
| 파일 | 무엇 |
|---|---|
| [docs/report.md](report.md) | 프로젝트 보고 본문(현재까지 전 과정·결과) |
| [docs/model-tuning-plan.md](model-tuning-plan.md) | CUDA 튜닝 스윕 설계서(Step 3에서 그대로 실행) |
| [docs/evaluation.md](evaluation.md) | 평가 방법론(지표·포화·슬라이스·상보성) |
| [docs/serving.md](serving.md) · [docs/serving-parity.md](serving-parity.md) | 서빙 경로·포맷 계약 검증 |
| `scripts/model_survey.py` · `scripts/hybrid_sweep.py` | 서베이·하이브리드 재현 스크립트 |
| `rag/datagen/clicklog.py` · `rag/datagen/ingest.py` | 클릭로그 클리닝·반입 |
| `rag/core/formatting.py` · `rag/modelprofile.py` | 포맷 패리티(불변 #1) |
