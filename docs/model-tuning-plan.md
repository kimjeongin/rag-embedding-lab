# 모델 튜닝·비교 설계서 (Qwen3-0.6B vs Nemotron-3-Embed-1B)

> 목적: "차기 임베딩 모델"을 **각 모델을 제대로 튜닝한 뒤** 공정하게 비교해 정한다.
> 단일 레시피 한 방(현재 상태)은 "제대로 학습했나"에 방어가 약하므로, 체계적 스윕으로
> 각 모델을 자기 최고점까지 끌어올린 근거를 남긴다. 이 문서는 **설계**이고, 실행은
> CUDA 서버에서 한다(무거운 1.14B 스윕 + 실환경 서빙 측정).

이 설계서 자체가 보고서의 **과제추진 사항 / 관련연구 벤치마킹** 항목이 된다.

---

## 0. 설계 원칙 (왜 이렇게 스윕하나)

랩의 스윕 도구(`lib/sweep`)는 이미 검증된 튜닝 방법론을 담고 있고, 그 위에서 설계한다:

- **한 번에 한 축(primary axis)만.** 결합 그리드 탐색이 아니라 단일 변수 프로브. 축을
  순차로 바꿔가며 승자를 이월하는 **좌표하강(coordinate descent)** — 제한된 컴퓨트에서
  Google Tuning Playbook이 권하는 방식.
- **비-LR 축은 LR과 동반 스윕(co-vary).** learning rate가 거의 모든 것과 상호작용하므로,
  loss·batch 같은 축을 고정 LR 하나로 판단하면 "그 LR에서만 나쁜 것"을 진짜 열등으로
  오인한다. 각 설정을 **자기 최적 LR에서** 판단한다(nuisance parameter 처리).
- **시드 반복으로 분산 측정.** 소량 데이터 FT는 시드마다 흔들린다. 같은 설정 × N시드의
  분산이 "실제 개선"과 "시드 노이즈"를 가르는 기준선이다.
- **median pruning.** 중간 성적이 나쁜 런을 조기 탈락시켜 컴퓨트를 아낀다(도구 내장).

---

## 1. 선행 게이트 — 측정기가 변별력이 있는가 (Phase 0)

**⚠️ 이걸 먼저 통과하지 못하면 아래 스윕은 전부 노이즈를 재는 것이다.**

현재 195문서 평가셋은 recall@10·@50이 포화(=1.0)라 변별력이 없다. nDCG@10·recall@1은
아직 여지가 있지만, **스윕 설정 간 차이가 시드 분산보다 작으면 아무 것도 못 고른다.**

- **Phase 0 실행:** 현재 최적 레시피(full/mnrl/b16/lr2e-5)를 **축 없음 × 5시드**로 돌려
  dev nDCG@10·recall@1의 **시드 표준편차**를 측정. 두 모델 각각.
- **판정:**
  - 시드 표준편차 ≪ 우리가 노리는 설정 간 차이(경험상 ≥0.01) → **게이트 통과, 스윕 진행.**
  - 시드 분산이 신호를 삼킴 → **스윕 전에 평가셋을 어렵게** 만든다(하드 네거티브 촘촘한
    distractor 추가, 또는 실 corpus 도입). 이게 향후계획 1순위와 연결된다.

> 보고 가치: "우리는 튜닝하기 전에 자를 먼저 검증했다"는 방법론적 rigor. 정직한 과정.

---

## 2. 스윕 단계 (각 모델 동일하게, 승자 이월)

영향 크고 상호작용 적은 축부터. 각 단계는 이전 단계 승자 위에서 진행한다.

| 단계 | 축 | 후보값 | 근거 |
|---|---|---|---|
| **P1** | learning_rate × batch_size | LR {5e-6, 1e-5, 2e-5, 5e-5} × batch {16, 32, 64} | 최적화의 핵심 두 노브. **CUDA에서 특히 중요** — batch↑ = MNRL의 in-batch negative↑라 보통 더 좋은데 MPS는 16에 묶여 있었다 |
| **P2** | loss | {mnrl, cached_mnrl, gist, triplet} × LR동반 | 대조학습 손실 선택. cached_mnrl은 같은 수식·저메모리라 큰 batch를 열어줌. **triplet은 hard negative 필수**(§3) |
| **P3** | dropout | {0, 0.1, 0.2} | 소량 데이터 과적합 방어 |
| **P4** | **Matryoshka** | on/off + dims (Nemotron [2048,1024,512,256], Qwen [1024,512,256,128]) | **이 과제의 핵심.** 차원=저장비용이므로, Matryoshka로 학습하면 Nemotron이 2048 정확도를 1024 저장비용에 근접시킬 수 있다 → "어떤 모델을, 어떤 차원으로"의 답 |
| **P5** | max_negatives | {0, 2, 4} (hard negative 채굴 후) | in-batch만 vs 채굴된 hard negative. §3 선행 |
| **P6** | method | full vs lora (lora_r {8,16,32}) | 학습 효율. 1.14B는 LoRA 이점이 큼. 서빙 품질 동등하면 학습비용 절감이 성과 |

**모델별 주의:**
- **Nemotron 1B:** `TRAIN_GRAD_CHECKPOINT=1` 유지(레시피 보존형 OOM 방어). CUDA VRAM이 크면
  꺼서 속도를 얻어도 되지만, 결과 비교를 위해 한 스윕 내에선 일관되게.
- **Qwen 0.6B:** 더 가벼워 batch를 더 키울 여지가 큼.

---

## 3. Hard negative 선행 채굴 (P2 triplet / P5 전제)

현재 `data-intranet/train.jsonl`은 negatives가 **0/741**이다. triplet loss와 max_negatives
스윕은 채굴이 있어야 의미가 있다.

- 랩의 margin-guarded 채굴(`datagen`)로 각 positive에 대해 "비슷하지만 오답"인 문서를
  채굴해 train.jsonl에 `negatives` 컬럼을 붙인다.
- 인트라넷 특성상 좋은 채굴원: 같은 시스템의 **다른 kind 페이지**(가이드 vs 장애공지),
  같은 kind의 **다른 시스템**(권한 페이지 27개 중 오답) — "비슷해서 틀리는" 쌍.
- 채굴 후 train 지문이 바뀌므로, **P1~P4는 채굴 전 셋으로, P5부터 채굴 셋으로** 진행하고
  두 셋을 레지스트리에서 지문으로 구분한다.

---

## 4. 평가 프로토콜 (공정성 계약)

- **주 지표: nDCG@10, recall@1.** recall@10·@50은 포화라 스윕 판단에 쓰지 않는다(관측만).
- **슬라이스 필수:** 표준 vs 은어. 전체 평균이 은어 붕괴를 가릴 수 있으므로 항상 분리.
- **유의성: paired sign-flip permutation.** 두 설정이 같은 쿼리를 풀었으므로 CI 겹침이 아니라
  쿼리별 델타로 p값. 시드 반복이 있으면 시드 평균 후 비교.
- **dev/final 규율:** 모든 스윕 선택은 **dev**로. 각 모델의 튜닝 승자가 정해지면 **final
  스플릿을 승자당 딱 1회** 돌려 확정(과적합 방지). final은 여러 번 보면 안 된다.
- **평가셋 지문 강제:** 지문이 다른 점수끼리는 비교 차단(레지스트리 내장).

---

## 5. 최종 대결 (모델 확정)

1. 각 모델의 P1~P6 dev 승자 확정 → 레시피 기록(train_meta.json 자동).
2. **두 승자를 final 스플릿에서 1회씩** 평가 → paired permutation으로 정확도 우열·유의성.
3. **rag-bench**로 두 승자의 실서빙 비용(지연 p50/p95/p99·GPU·저장·색인) 측정 —
   **CUDA에서** 재측정(MPS 수치는 이전 안 됨). 정확도가 유의하게 갈리면 그걸로 결정,
   무의미하면 비용으로 결정(현 서비스 환경에선 비용차가 작아 사실상 정확도가 결정).
4. Matryoshka 승자가 있으면 **차원별 정확도-저장비용 곡선**을 같이 제시 — "2048이냐
   1024냐"까지 답.

---

## 6. CUDA에서 실행하는 법

**방법 A — 랩 UI(권장, 방법론 내장).** CUDA 서버에서 `make run`(또는 `uv run rag-serve`)로
서버를 띄우고 포트포워딩해 Train 탭 → 스윕 모드. median pruning·keep_top_k·라이브
리더보드·학습 후 자동 평가가 전부 붙는다. 축·값·LR동반·시드를 UI에서 지정하면
미리보기가 실행될 런 목록을 그대로 보여준다.

**방법 B — 헤드리스 CLI(서버 없이).** 한 단계를 직접 돌릴 때. 예) P1의 한 점:
```bash
TRAIN_BASE_MODEL="nvidia/Nemotron-3-Embed-1B-BF16" \
TRAIN_OUTPUT_DIR="outputs/nemo-sweep" \
TRAIN_FILE=data-intranet/train.jsonl TRAIN_EVAL_FILE=data-intranet/test.jsonl \
TRAIN_LR=2e-5 TRAIN_BATCH_SIZE=32 TRAIN_LOSS=mnrl TRAIN_GRAD_CHECKPOINT=1 \
uv run rag-train
# 학습 후:
EVAL_DIR=data-intranet/eval ST_MODEL=outputs/nemo-sweep-mnrl-e{N} uv run rag-eval
```
전 축 env 노브: `TRAIN_LR · TRAIN_BATCH_SIZE · TRAIN_LOSS · TRAIN_DROPOUT ·
TRAIN_MATRYOSHKA(+_DIMS) · TRAIN_MAX_NEGATIVES · TRAIN_METHOD · TRAIN_LORA_R ·
TRAIN_SEED · TRAIN_MODEL_PROFILE`. (UI 스윕이 이걸 자동 조합해준다 — 방법 A 권장.)

---

## 7. 보고서에 남길 것 (6단 프레임 매핑)

- **관련연구 벤치마킹:** 튜닝 방법론(좌표하강·LR 동반·median pruning)의 출처와 손실 4종·
  Matryoshka·hard negative의 이론 근거.
- **과제추진 사항:** Phase 0 게이트 → 단계별 스윕 → 최종 대결의 **설계·실행·분석** 전 과정.
  특히 "자를 먼저 검증하고 튜닝했다"는 순서.
- **과제성과/기대효과:** 각 모델의 튜닝 전→후 향상폭, 차기 모델 결정과 그 근거(정확도
  유의성 + 비용 곡선), Matryoshka 차원-비용 트레이드오프.
- **향후계획:** 실 corpus/로그 도입 시 재검증, 하이브리드 융합 튜닝(별도 성과 경로).
