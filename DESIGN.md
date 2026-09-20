# 설계: 핵수용체 특화 Ligand-Receptor 양방향 검색 프레임워크

BPA 전용 모델을 새로 학습시키는 대신, 기존에 검증된 모델들(DrugCLIP, Uni-Mol Docking V2, IGModel)을 그대로 이어 붙이고, 딱 한 곳(hard-negative 파인튜닝)만 새로 학습시키는 설계. 배경은 `README.md`와 `benchmark_db/`(연구실 선배의 9/10~9/14 실험 기록)에 있음 — 아래는 그 교훈을 반영한 다음 단계 설계.

## 목표

> **핵수용체(nuclear receptor) 패밀리에 특화된 사람 대상 ligand-receptor 양방향 검색 프레임워크를 구축하고, BPA/비스페놀류를 플래그십 검증 사례로 엄밀하게 적용한다.**

- ❌ Cross-species 확장
- ❌ 새 ligand/receptor encoder 아키텍처 설계
- ❌ Teacher(도킹) → student(retrieval) knowledge distillation — 9/10·9/13에서 이미 실패 확인됨
- ❌ "BPA만으로" 파인튜닝 — 9/11에서 데이터 부족으로 일반화 실패 확인됨 (297건으로 새 리셉터 AUROC 0.540→0.497)
- ✅ 양방향 검색(이미 구현·검증됨)
- ✅ Hard-negative 파인튜닝 — 단, BPA 하나가 아니라 EDC 관련 핵수용체 패밀리 전체 스케일로
- ✅ 포즈 + 결합력 검증 단계 (기존 모델 그대로)

## 왜 이렇게 스코프를 잡았나

9/11 실험이 핵심 근거: **파인튜닝 데이터의 도메인 안에서는 실제로 좋아졌음** (새 논문 AUROC 0.623→0.856, 새 scaffold 0.623→0.785). 실패한 건 "완전히 학습에 없던 리셉터"로의 일반화였지, "도메인 내 특화" 자체가 아니었다. 그러므로 BPA 하나가 아니라 **BPA가 속한 EDC/핵수용체 도메인 전체**를 학습 범위에 포함시키면, 같은 방법론이지만 실패하지 않을 규모가 될 가능성이 높다.

포즈 쪽(9/10, 9/13)은 반대로 데이터 양과 무관하게 실패했다 — "거리만 입력받는 DrugCLIP류 인코더는 원자 반사/방향 관계를 표현할 방법이 없다"는 구조적 한계였기 때문. 그래서 포즈는 파인튜닝하지 않고, 이미 대규모(PDBbind ~1.9만 복합체)로 학습되어 새 리간드에 바로 일반화되는 기존 모델(Uni-Mol Docking V2, IGModel)을 zero-shot으로 그대로 쓴다.

## 전체 구조

```
[1단계] 검색 엔진 — DrugCLIP (기존 pretrained 포켓/분자 인코더 그대로)
        + hard-negative 파인튜닝 (핵수용체 패밀리 한정, 새 데이터로 추가 학습만)
        → ligand ↔ receptor 양방향 랭킹

              ↓ top-k만

[2단계] 포즈 — Uni-Mol Docking V2 (zero-shot, 재학습 없음)
        → 실제 3D 포즈

              ↓

[3단계] 재채점 — IGModel (zero-shot, 재학습 없음)
        → 포즈 신뢰도(RMSD) + 결합력(pKd)
```

새로 "학습"하는 건 1단계의 hard-negative 파인튜닝 하나뿐. 2, 3단계는 이미 설치·검증 끝난 것을 그대로 붙인다.

## 1. 검색 엔진 (DrugCLIP + hard-negative 파인튜닝)

**베이스**: 기존 `DrugCLIP/checkpoint_best.pt` (포켓/분자 인코더), 새 아키텍처 없음.

**리셉터 범위**: 사람 핵수용체 슈퍼패밀리 중 EDC 문헌에서 실제 타깃으로 보고된 것 위주로 시작 — ERα, ERβ, ERRγ, PPARγ, AR, THRα/β, PR, GR, VDR, PXR 등. (이미 `bpa_panel/receptors.csv`에 7개 있음, 확장 필요)

**데이터 소스**:

| 소스 | 용도 | 현황 |
|---|---|---|
| ChEMBL | 화합물-리셉터 실측 IC50/Ki | `benchmark_db/bpa_binding_supervision_20260911/`에서 이미 297개 뽑은 파이프라인 있음, 범위만 확장 |
| Tox21 / ToxCast | EDC 특화 대규모 qHTS assay (화합물 수천 개 × 이 리셉터들) | **확인 완료 (아래 표), 실제 pull은 아직 안 함** |
| PDBbind | 포켓 구조 정의용 | 이미 사용 중 |

**Hard negative 구성**: 같은 리셉터 패밀리 내 다른 specificity를 서로의 negative로 사용 (ERα-ERβ-ERRγ 서로, AR-PR-GR 서로 등).

**실제 사례로 확인된 문제 (BPA, 기존 7개 리셉터 패널 기준) — README 8단계에서 포켓 풀링 버그 수정 후 갱신됨**:

| 리셉터 | DrugCLIP 점수 (수정 전 → 후) | 분류 | 비고 |
|---|---|---|---|
| ERα | 0.5711 → **0.8440** | positive (Delfosse 2012, 결정구조 3UU7) | 포켓 버그 영향받았던 리셉터, 점수 크게 상승 |
| ERRγ | 0.8267 (변화 없음) | positive (Matsushima 2007, Kd=5.5nM) | 원래부터 영향 없는 리셉터 (카피 1개) |
| ERβ | 0.4183 (변화 없음) | positive (Delfosse 2012) | 영향 없는 리셉터 |
| AR | 0.5887 (변화 없음) | hard negative (직접결합 결정구조 없음, DHT 구조 대리) | 영향 없는 리셉터 — ERα가 0.8440으로 올라오면서 **"AR이 ERα보다 높다"는 false positive는 사라짐** (0.5887 < 0.8440) |
| THRβ | 0.3707 (변화 없음) | hard negative (간접 기전만 보고) | 영향 없는 리셉터 |
| PR | 0.3497 → **0.5523** | hard negative (유도체 BPAF 도킹 연구만 존재) | 포켓 버그 영향받았던 리셉터 |
| PPARγ | 0.0148 → **0.5381** | positive (결정구조 9F7W 존재) | 포켓 버그 영향받았던 리셉터 — **false negative였던 게 순수 버그였음이 확인됨** |

**교훈**: PPARγ와 AR 둘 다 "모델이 이상하다"로 보였지만, 실제로는 **하나는 우리 쪽 데이터 버그(PPARγ), 하나는 애초에 버그와 무관한 진짜 관찰(AR — 다만 수정 후엔 ERα와의 격차가 커져서 "hard negative치고 너무 높다"는 정도로 약해짐)**이었다. 모델을 탓하기 전에 항상 입력(포켓 정의)부터 의심할 것. AR을 hard-negative 파인튜닝 대상에 넣을지는 이 격차(0.5887 vs 0.8440)가 실제로 파인튜닝이 필요할 만큼 큰지 재판단 필요.

**목표 규모**: 9/11의 337건보다 최소 한 자릿수 이상 필요 — **PubChem BioAssay API로 실제 확인 완료, 충분함**:

| 리셉터 | Tox21 AID | 총 화합물 수 | Active / Inactive / Inconclusive |
|---|---|---|---|
| ERα (antagonist) | 743078 | 10,486 | 433 / 7,996 / 2,057 |
| AR (agonist) | 743053 | 10,486 | 372 / 9,070 / 1,044 |
| AR (antagonist) | 743063 | 10,486 | 670 / 7,770 / 2,046 |
| PPARγ (antagonist) | 743191 | 9,305 | 174 / 5,202 / 3,929 |
| ERRγ (agonist) | 1224820 | 1,366 | (소규모, 별도 확인용) |

Tox21은 리셉터마다 다른 화합물 세트가 아니라 **거의 같은 ~1만 개 화합물 라이브러리를 여러 리셉터 assay에 반복 스크리닝**한 구조라, ERβ/THRβ/PR/GR/VDR/PXR도 비슷한 규모(~9천~1만 개)로 있을 가능성이 높음 (개별 확인은 남음). 9/11의 297개와는 자릿수가 다름 — 9/11이 놓쳤던 "새 리셉터 일반화" 실패를 극복할 만한 규모.

**BPA/비스페놀류 제외 확인**: ERα antagonist assay(743078)에서 우리 패널 11개 화합물 중 10개가 실제로 테스트돼 있었음:

| 화합물 | CID | Tox21 결과 |
|---|---|---|
| BPA | 6623 | Active |
| BPAF | 73864 | Active |
| BPB | 66166 | Active |
| BPZ | 232446 | Active |
| BPC | 6620 | Active |
| TCBPA | 6619 | Active |
| BPS | 6626 | Inactive |
| BPF | 12111 | Inactive |
| BPE | 608116 | Inconclusive |
| TBBPA | 6618 | Inconclusive |
| BPAP | 623849 | 테스트 안 됨 |

**→ hard-negative 파인튜닝 데이터를 Tox21/ChEMBL에서 뽑을 때, 이 11개 CID는 전부 제외하고 최종 BPA 케이스 스터디 검증용으로만 남겨둔다** (요청에 따라 확정).

## 2. 포즈 (Uni-Mol Docking V2, 변경 없음)

- 이미 설치·검증 완료 (`UniMolDockingV2/`)
- BPA 결정구조 3곳 대비 대칭 보정 RMSD 0.55~0.76Å로 검증됨 (`bpa_panel/docking/verify_bpa/`)
- 자체 확신도(prmsd_score)는 대칭 분자에 오염되므로, **평가 시 항상 대칭 보정 RMSD와 함께 봐야 함** (이번에 발견한 버그, 재발 방지용으로 명시)

## 3. 재채점 (IGModel, 신규 도입)

포즈를 새로 만드는 모델이 아니라, **이미 있는 후보 포즈를 평가하는(rescoring) 모델**. DrugCLIP처럼 포켓/리간드를 따로 인코딩해서 비교하는 게 아니라, 포켓-리간드를 하나의 상호작용 그래프로 묶어 GNN에 넣는다 — 원자 간 상대 위치를 애초부터 표현할 수 있는 구조.

- 입력: 단백질 PDB + 참조 리간드(포켓 정의용) + Uni-Mol Docking V2가 만든 후보 포즈
- 출력: RMSD 추정치(포즈 신뢰도) + pKd(결합력) — 둘 다 PDBbind 실측 결정구조/실측 Kd·Ki·IC50 기반으로 학습됨 (RMSD 라벨은 결정구조 상 진짜 위치 대비 거리, pKd는 같은 복합체의 실측 결합값)
- CASF-2016 벤치마크에서 Top1 도킹 포즈 선택 성공률 97.5% — 선배가 9/13, 9/14에서 겪은 "여러 후보 중 1등 고르기 실패" 문제를 정면으로 겨냥한 벤치마크
- 가중치 2종 제공: 참조 리간드 기반(`saved_model.pth`, 우리 상황에 맞음) / self-pose-ref 기반(참조 리간드 없이 blind하게 쓸 때)
- 아직 미설치 — 다음 작업

## 4. 평가

- Recall@k, AUROC — **도메인 내(known family) vs 완전 unseen 리셉터를 분리해서 보고** (9/11에서 얻은 교훈: pooled 지표만 보면 unseen 실패를 놓침)
- Scaffold split, homology-aware split
- 포즈: PoseBusters 기준 + 대칭 보정 RMSD 필수

## 5. BPA 케이스 스터디 (기존 결과 재사용 + 비교)

- 기존 `bpa_panel/` 전체(11개 화합물 × 7개 리셉터, 문헌 검증, 넓은 라이브러리 재검증)는 그대로 유지
- **hard-negative 파인튜닝 전/후로 같은 패널을 다시 돌려서 비교** — 파인튜닝이 실제로 뭘 개선했는지 보여주는 핵심 결과

## MVP 순서

1. ~~Tox21/ToxCast에서 이 리셉터 패밀리 관련 실측 데이터 규모 확인~~ **완료** — ERα/AR/PPARγ 각 9천~1만 건, 9/11(297건) 대비 충분한 규모 확인. BPA류 11개 화합물의 실제 CID·결과도 확인, 제외 리스트 확정.
2. ChEMBL + Tox21 합쳐서 hard-negative 파인튜닝 데이터셋 구축 (**위 11개 BPA/비스페놀류 CID는 전부 제외** — BPA는 최종 validation 전용으로 남겨둠)
3. DrugCLIP hard-negative 파인튜닝 (9/11 방식을 도메인 전체로 확장)
4. 도메인 내 held-out 리셉터 평가 + 완전 unseen 리셉터 평가를 분리해서 보고
5. BPA 패널 재실행, 파인튜닝 전/후 비교
6. IGModel 설치, Uni-Mol Docking V2 출력 재채점에 연결

## 참고

- 이 설계에 반영된 실증적 교훈의 원본 기록: `benchmark_db/bpa_conformer_pilot_20260910/`, `benchmark_db/bpa_binding_supervision_20260911/`, `benchmark_db/bpa_pose_embedding_alignment_20260913/`, `benchmark_db/bpa_receptor_pose_comparison_20260914/`
- 전체 진행 과정: `README.md`
