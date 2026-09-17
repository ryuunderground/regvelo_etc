# DrugCLIP으로 BPA/비스페놀류 내분비교란물질 스크리닝

DrugCLIP(단백질 포켓-화합물 대조학습 기반 초고속 가상 스크리닝 모델, Science 2026)을 로컬(CPU, Apple Silicon)에서 돌려서, 비스페놀A(BPA)와 그 유사체들이 어떤 핵수용체에 얼마나 잘 붙는지 스크리닝하고, 실제 문헌/결정구조와 대조 검증한 기록입니다. 아래는 처음부터 지금까지 실제로 한 작업을 순서대로 정리한 것입니다.

## 왜 이걸 하는가

BPA는 잘 알려진 내분비교란물질(EDC)입니다. "BPA가 어떤 리셉터에 잘 붙는가"와 "BPA 유사체(대체 화학물질로 쓰이는 BPS, BPF 등)도 마찬가지로 위험한가"를 계산적으로 스크리닝하고, 그 결과가 실제 생물학/문헌과 맞는지를 단계마다 의심하고 검증하면서 진행했습니다.

---

## 1단계 — DrugCLIP 환경 구축 (CPU, uv)

**논문**: `papers/drugclip.pdf` — 포켓 인코더와 분자 인코더를 대조학습으로 정렬해서, 결합 예측을 "회귀"가 아니라 "임베딩 유사도 검색" 문제로 푸는 모델. 도킹보다 최대 1000만 배 빠름.

- `github.com/THU-ATOM/DrugCLIP` (NeurIPS 2023 공식 코드)를 `DrugCLIP/`에 클론
- `uv`로 Python 3.10 가상환경(`.venv/`) 생성, `requirements.txt`로 의존성 설치 (torch 2.0.1, rdkit==2022.9.5, biopandas 등)
- `unicore`(모델 프레임워크, DrugCLIP과 같은 랩 제작)를 소스에서 빌드 — `pkg_resources`/`setuptools` 버전 문제 등 몇 가지 빌드 이슈 해결
- **CPU 전용 버그 2개 발견 및 패치** (원본 코드는 GPU 전제로 작성됨):
  - `unicore.utils.move_to_cuda`가 CPU에서 무조건 `torch.cuda.current_device()`를 호출해서 크래시 → CPU일 때 no-op으로 패치
  - `unimol/models/drugclip.py`의 `logit_scale` 파라미터가 `device="cuda"`로 하드코딩되어 있어 모델 생성 자체가 CPU에서 실패 → 제거
  - `retrieval.py`가 검색 결과를 계산만 하고 출력/저장을 안 하는 버그도 발견해서 고침 (원본 저장소 자체의 버그)
- 모델 가중치(`checkpoint_best.pt`, 1.1GB)와 예제 데이터(`mols.lmdb` 294만 화합물, `pocket.lmdb`)를 Google Drive에서 다운로드 — 중간에 네트워크가 끊겨서 재다운로드하기도 함
- 300개 화합물 서브셋으로 **end-to-end 스모크 테스트 성공**

## 2단계 — 역방향 검색(target fishing) 추가

기존 DrugCLIP은 "포켓 하나 고정 → 화합물 랭킹"만 지원. 반대 방향("화합물 하나 고정 → 어떤 리셉터에 잘 붙는지 랭킹")은 없어서 직접 추가:

- `unimol/tasks/drugclip.py`에 `encode_pockets_once()` / `retrieve_pockets()` 추가 (기존 `retrieve_mols()`의 미러 버전, 리간드-포켓 역할만 바꿈)
- `unimol/retrieve_receptors.py` 새 CLI 진입점
- 검증: 정방향 1위 화합물의 점수(0.5582)와 역방향 결과가 정확히 일치 — 구현이 맞다는 교차검증

## 3단계 — BPA/비스페놀 패널 구축 (11개 화합물 × 7개 리셉터)

**중요**: 여기서부터는 기존 DrugCLIP 체크포인트를 **그대로 쓰는 조회(zero-shot)** 이지, 재학습이 아닙니다. 화합물 2~3개로 파인튜닝하면 통계적으로 의미가 없다는 게 이 프로젝트 초반에 나온 핵심 결론이라, 계속 이 원칙을 지켰습니다.

- **화합물 11개** (`bpa_panel/compounds.csv`): BPA + BPS, BPF, BPAF, BPB, BPE, BPZ, BPC, BPAP, TBBPA, TCBPA. SMILES는 전부 PubChem API에서 직접 조회해서 검증 (기억에 의존 안 함)
  - 주의사항 하나 발견: PubChem이 "Bisphenol C"라고 부르는 화합물(CID 6620)은 독성학 문헌이 실제로 말하는 BPC(다른 구조)와 다른 화합물이었음 — CSV에 명시
- **리셉터 7개** (`bpa_panel/receptors.csv`): ERalpha, ERbeta, ERRgamma, PPARgamma, AR, THRbeta, PR — 문헌상 BPA 타깃으로 보고된 핵수용체들. 전부 **실제 PDB 결정구조**(AlphaFold 아님) 사용, 각각 출처 논문/PDB ID 기재
- `build_pockets.py`: 실제 PDB 다운로드 후 포켓 추출 (참조리간드 6Å 이내 잔기, 원본 저장소의 `write_dude_multi.py` 로직 재사용)
- `build_mols.py`: RDKit으로 11개 화합물의 3D 컨포머 생성
- `score_matrix.py` (신규 CLI): 기존 `retrieve_mols`/`retrieve_pockets`는 top-k만 반환하는데, 7×11처럼 작은 패널은 전체 쌍을 다 보는 게 낫다고 판단해서 **전체 매트릭스**를 반환하는 버전을 추가
- **결과**: BPA가 ERRgamma에서 11개 중 2위(상위 90.9%) — 문헌상 알려진 가장 강한 친화도(Kd=5.5nM)와 일치. ERalpha에서는 중간(상위 63.6%)

## 4단계 — 문헌 검증 (병렬 에이전트 4개)

ERalpha에서 BPA보다 점수가 높게 나온 화합물 4개(BPZ, BPB, BPC, TCBPA)에 대해, 각각 "진짜 문헌에도 그런 근거가 있는지" 에이전트 4개를 병렬로 띄워서 확인 (`bpa_panel/eralpha_literature_check.csv`):

| 화합물 | 결과 |
|---|---|
| BPZ | ✅ 확인됨 (2개 독립 연구, BPA 대비 ~3배 강함) |
| BPB | ✅ 확인됨 (15개 연구 리뷰 + 직접비교) |
| BPC | ⚠️ **검증 불가** — 우리가 스크리닝한 화합물이 문헌이 말하는 "진짜 BPC"와 다른 구조였음 |
| TCBPA | 🟡 문헌 엇갈림 — 모델 예측을 뒷받침 못함 |

모델 예측을 맹신하면 안 되고, 특히 이름이 중의적인 화합물은 구조 identity부터 재확인해야 한다는 교훈.

## 5단계 — "친구들만 비교한 거 아니냐" 재검증

사용자 지적: 11개 패널이 전부 비스페놀류라서, "BPA가 상위 63.6%"라는 숫자는 **BPA의 사촌들 사이에서의 순위**일 뿐 전체 화학공간에서의 순위가 아님. 실제로 검증:

- DrugCLIP의 실제 화합물 라이브러리(294만 개)에서 2만 개를 랜덤 샘플링 + BPA 추가해서 ERalpha 기준 전체 재랭킹 (`bpa_panel/library_ranking.csv`)
- **결과: BPA가 20,001개 중 3위 (상위 0.01%)**
- 근데 이것도 다시 의심해봄: "혹시 2만 개 샘플에 BPA의 경쟁자(다른 비스페놀류)가 하나도 없어서 그런 거 아니냐?"
- 294만 개 전체를 SMARTS 구조 검색으로 훑어본 결과, **비스페놀류처럼 생긴 화합물이 전체에서 14개뿐** (`bpa_panel/library_composition_check.csv`). 즉 "3위"는 "BPA가 최강"이 아니라 "이 라이브러리엔 BPA의 경쟁자가 거의 없다"는 뜻에 더 가까움 — 정직하게 정정해서 기록

## 6단계 — 포즈 예측 모델 추가 (Uni-Mol Docking V2)

DrugCLIP은 리간드-포켓을 벡터 하나씩으로 압축해서 코사인 유사도만 비교하는 구조라, **애초에 "어떤 모양으로 붙는지"(포즈)를 표현할 방법이 없음** — 파인튜닝으로 해결 안 되는 아키텍처 한계. 그래서 포즈 예측은 별도의 사전학습 모델을 붙이기로 결정:

- **Uni-Mol Docking V2** 선택 (DiffDock 대신) — DrugCLIP과 같은 랩(dptech-corp) 제작, PoseBusters 벤치마크 정확도 더 높음(77.6%), blind docking(결합부위 안 알려줘도 됨)
- `UniMolDockingV2/`에 설치, 체크포인트(487MB) 다운로드
- **패치**: 모델이 내부적으로 conf_size(10)개 후보 포즈 중 최선을 고를 때 쓰는 자체 신뢰도 점수(`prmsd_score`, predicted RMSD)를 계산만 하고 버리길래, 결과 파일/CSV에 남도록 수정
- **버그 발견 및 수정**: ERalpha/PPARgamma/PR 3개 리셉터의 PDB 파일에 참조리간드가 여러 카피 들어있어서, 그냥 다 풀링하면 도킹 박스가 여러 결합부위에 걸쳐 잘못 커지는 문제 → 하나의 인스턴스만 쓰도록 수정
- **36쌍 테스트** (`bpa_panel/docking/`): 리셉터 6개(BPA 잘 붙는 3개 + 안 붙는 3개) × 리간드 6개(비스페놀 3개 + 무관한 랜덤 화합물 3개)
  - 결과: 비스페놀류가 리셉터 상관없이 일관되게 랜덤 화합물보다 자체 신뢰도가 낮음 (평균 12.3 vs 7.8)

## 7단계 — "신뢰도 낮음"의 진짜 원인: 대칭성 버그 발견

사용자 지적: "그 신뢰도가 이미 10개 후보 중 최선을 고른 거면, 더 해봐야 소용없는 거 아니냐?" (맞음, 코드로 확인) + "우리 연구실에서 BPA가 ERalpha에 잘 붙는다고 나왔는데?" → 실제 정답(결정구조)과 직접 비교해보기로 함:

- ERalpha(3UU7)/ERRgamma(2E2R)/PPARgamma(9F7W)는 **실제로 BPA가 결합된 결정구조**라서 정답을 알고 있음
- BPA를 이 3곳에 도킹시켜서, 예측 포즈를 진짜 결정구조상의 BPA 위치와 비교 (`bpa_panel/docking/verify_bpa/`)
- **핵심 발견**: 원자 순서 그대로 비교한 RMSD는 2~3.7Å(그저 그럼)인데, **BPA의 좌우 대칭성을 고려해서 비교(symmetry-aware RMSD)하면 전부 1Å 미만** (0.55~0.76Å — PoseBusters 기준 "좋은 포즈" 안쪽)

| 리셉터 | 단순 RMSD | 대칭 보정 RMSD | 모델 자체 확신도 |
|---|---|---|---|
| ERalpha | 3.71Å | **0.55Å** | 12.6 (나쁨) |
| ERRgamma | 3.37Å | **0.55Å** | 11.8 (나쁨) |
| PPARgamma | 2.05Å | **0.76Å** | 12.2 (나쁨) |

**결론**: 모델이 예측한 포즈는 실제로 거의 정확했는데, 모델 자체 확신도 계산 방식이 BPA처럼 좌우 대칭인 분자(고리 2개가 거의 똑같이 생김)를 "다른 포즈"로 착각해서 확신도를 실제보다 훨씬 나쁘게 평가하고 있었던 것. 6단계에서 본 "DrugCLIP 점수와 도킹 신뢰도가 안 맞는다"는 문제도, 두 모델이 실제로 어긋난 게 아니라 **신뢰도 지표 자체가 대칭 분자에 불리하게 오염되어 있었던** 것으로 정리됨. 연구실에서 확인한 "BPA-ERalpha 잘 붙음"이 DrugCLIP 랭킹 + 실제 결정구조 + 이번 도킹 재현까지 세 겹으로 뒷받침됨.

---

## 폴더 구조

```
drugclip/
├── DrugCLIP/                  # THU-ATOM/DrugCLIP 원본 코드 + CPU 패치 (1~2단계)
│   ├── unimol/retrieval.py            # 정방향 검색 (포켓→화합물)
│   ├── unimol/retrieve_receptors.py   # 역방향 검색 (화합물→리셉터)
│   └── unimol/score_matrix.py         # 전체 매트릭스 (소규모 패널용)
├── UniMolDockingV2/            # 포즈 예측 모델 (6단계)
├── bpa_panel/                  # BPA/비스페놀 스크리닝 결과 전체 (3~7단계)
│   ├── compounds.csv / receptors.csv          # 화합물/리셉터 패널 + 출처
│   ├── score_matrix.csv / results_ranked.csv   # 7×11 결합 스코어
│   ├── eralpha_literature_check.csv            # 문헌 검증 결과
│   ├── library_ranking.csv / library_composition_check.csv  # 넓은 라이브러리 재검증
│   └── docking/                                 # 포즈 예측 결과
│       ├── inputs/, predict_sdf/                   # 36쌍 도킹 입출력
│       ├── docking_summary.csv                     # DrugCLIP 점수 + 도킹 신뢰도 통합
│       └── verify_bpa/                             # 결정구조 대비 검증 (7단계)
└── requirements.txt
```

## 재현 방법 (요약)

1. `uv venv --python 3.10 .venv && source .venv/bin/activate && uv pip install -r requirements.txt`
2. `unicore`를 소스에서 빌드: `uv pip install "git+https://github.com/dptech-corp/Uni-Core.git" --no-build-isolation`
3. `DrugCLIP/checkpoint_best.pt`는 [원본 레포 Google Drive](https://github.com/THU-ATOM/DrugCLIP)에서 받아야 함 (용량 문제로 이 레포엔 커밋 안 함)
4. `bpa_panel/`, `bpa_panel/docking/`의 각 `build_*.py` 스크립트를 순서대로 실행하면 전체 파이프라인 재현 가능

## 알려진 한계 / 다음에 할 일

- `build_pockets.py`(3단계, `score_matrix.csv`용 포켓 추출)에는 7단계에서 발견한 "리간드 여러 카피 풀링" 버그가 아직 남아있음 (ERalpha/PPARgamma/PR 3개 리셉터) — 도킹 쪽(`build_docking_inputs.py`)은 고쳤지만 스크리닝 점수 쪽은 재작업 안 함
- BPS/BPAF/BPZ의 포즈 정확도는 대칭성 문제로 자체 확신도만으로는 판단 불가하고, 이 3개는 결정구조가 없어서 직접 검증도 못 함 — BPA와 같은 계열이라 마찬가지로 정확할 것이라는 추론만 가능
- "구조와 결합력을 동시에 보는 모델 하나"를 원한다면, DrugCLIP을 파인튜닝하는 게 아니라 포즈를 입력받아 재채점하는 별도의 pose-aware scoring 모델(GNINA, RTMScore 계열)을 3번째 단계로 추가하는 게 다음 방향으로 논의됨 (아직 미착수)
