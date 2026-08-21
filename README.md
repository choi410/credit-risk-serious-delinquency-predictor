# 향후 2년 내 심각한 연체 위험 CSV 예측 프로그램

고객 신용·재무정보 CSV를 입력하면 고객별 연체 위험 점수와 고객군을 계산하고 CSV 및 군집 이미지를 생성합니다.

```text
result/
├── 00_결과요약.csv
├── 01_전체고객_위험순위.csv
├── 02_추가심사대상.csv
├── 03_1차기준통과.csv
├── 04_고객별_군집결과.csv
├── 05_군집별_특성요약.csv
├── 06_학습데이터_군집별연체율.csv
└── 07_클러스터링_모델비교.csv

figures/
├── 01_군집분포_PCA.png
├── 02_군집별_특성_히트맵.png
├── 03_군집별_고객수.png
├── 04_군집별_실제연체율.png
├── 05_군집신뢰도_분포.png
└── 06_군집수_선정지표.png
```

## 판정 기준

- `RiskScore >= 0.074241900037`: `대출 보류(추가 심사)`
- `RiskScore < 0.074241900037`: `1차 기준 통과`

임계값은 개발 데이터의 5-fold OOF 예측에서 Recall 0.75 이상을 만족하는 후보 중 Precision이 가장 높은 값입니다. 법적 대출 승인·거절 기준이나 정확한 7.42% 연체확률을 뜻하지 않습니다.

## 가장 간단한 실행

1. 예측할 CSV를 `input/customer_data.csv`로 저장합니다.
2. `run_prediction.bat`을 실행합니다.
3. `result` 폴더의 CSV와 `figures` 폴더의 군집 이미지를 확인합니다.

직접 명령어를 사용할 수도 있습니다.

```powershell
python predict_csv.py --input "입력파일.csv" --output-dir "result"
```

## 필요한 입력 컬럼

```text
RevolvingUtilizationOfUnsecuredLines
age
NumberOfTime30-59DaysPastDueNotWorse
DebtRatio
MonthlyIncome
NumberOfOpenCreditLinesAndLoans
NumberOfTimes90DaysLate
NumberRealEstateLoansOrLines
NumberOfTime60-89DaysPastDueNotWorse
NumberOfDependents
```

`Id` 또는 `Unnamed: 0` 컬럼이 있으면 고객 ID로 사용합니다. 두 컬럼이 모두 없으면 1부터 순서대로 ID를 생성합니다. `MonthlyIncome`과 `NumberOfDependents`의 빈 값은 학습 Pipeline에서 처리합니다.

## 모델 정보

- 모델: Histogram Gradient Boosting
- 전처리: 연체 특수값 96·98 처리, `age=0` 처리, 결측·특수값 indicator 3개, 중앙값 대체
- 불균형 처리: 없음
- 확인용 holdout ROC-AUC: 0.8684
- 확인용 holdout PR-AUC: 0.4102
- 확인용 holdout Recall: 0.7546
- 확인용 holdout Precision: 0.2291

## 고객 클러스터링

- 비교 모델: K-Means, Gaussian Mixture Model(GMM)
- 후보 군집 수: 2~6개
- 최종 모델: 고객별 소속 확률을 제공하는 GMM
- 최종 군집 수: 수렴하고 최소 군집 비율 2% 이상인 GMM 후보 중 BIC가 가장 낮은 값
- 전처리: 특수값·결측 처리 → 학습 분위수 clipping → log1p → RobustScaler
- 군집 학습 제외 컬럼: `Id`, `SeriousDlqin2yrs`
- 군집 해석: 군집 학습이 끝난 뒤에만 군집별 실제 연체율 비교

`ClusterID`는 고객 유형을 뜻하며 대출 승인·거절 등급이 아닙니다. PCA 좌표는 시각화용이며 실제 군집화는 전처리된 전체 피처를 사용합니다.

군집 결과는 별도의 고객 유형 분석으로 제공합니다. 기존 HGB 위험점수와 검증된 임계값을 바꾸지 않기 위해 현재 분류모델의 입력 피처로는 사용하지 않습니다.

Precision이 약 22.9%이므로 `추가 심사 대상`을 실제 대출 거절 고객으로 확정하면 안 됩니다. 이 프로그램의 결과는 추가 심사 순서를 정하기 위한 모델 위험 점수입니다.

## 모델 재학습

학습 데이터를 변경한 경우에만 실행합니다.

```powershell
python train_and_export_model.py --data-dir "학습데이터폴더경로"
python train_clustering.py --data-dir "학습데이터폴더경로"
```

일반적인 예측 실행에는 재학습이 필요하지 않습니다.
