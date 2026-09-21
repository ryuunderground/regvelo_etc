슈뢰딩거 점수 모음

범위: bpa_pose_embedding_alignment_20260913/flexible_five_v1에서 생성하고 bpa_ab_capacity_ablation_20260921에서 사용한 자료.
all_learning_scores_and_predictions.csv: 4,619개 모양 대표행; split은 train/validation/test/별도 진단을 구분. teacher_score는 같은 내부모양 그룹의 원본 docking score 중앙값. glide_gscore, epik_state_penalty, glide_emodel은 대표 포즈의 원본 값이므로 그룹 teacher_score와 직접 합산 비교하면 안 됨. member_teacher_scores에 그룹 구성원의 원본 점수.
all_original_pose_scores.csv: QC를 통과한 묶기 전 4,778포즈; teacher_score는 원본 r_i_docking_score.
bpa_raw_glide_components.csv: BPA 다섯 포켓 100포즈의 maegz 원본 상세 속성.
original_glide_csv/: ESR1 ESR2 ESRRG AR PXR VDR의 원본 flexible_sp.csv. 이는 학습에 남은 모든 포즈를 펼친 표와 행 단위가 다를 수 있으므로 전체 포즈 비교에는 all_original_pose_scores.csv 사용.

점수는 낮을수록 좋은 방향. Emodel과 docking score는 별도 지표이며 같은 값이 아님. 이는 실험 결합 친화도가 아닌 계산 점수.
VDR는 reference redocking QC로 주 학습/검증에서 제외되어 별도 진단으로 보관됨. BPA 역시 학습/모델선택에서 제외된 진단 자료.
source_pose_file_index는 maegz 내 0부터 시작하는 위치. NR1I2는 PXR의 유전자명.
