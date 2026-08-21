# 향후 2년 내 심각한 연체 위험 CSV 예측 프로그램

고객 신용·재무정보 CSV를 입력하면 고객별 모델 위험 점수를 계산하고 다음 네 파일을 생성합니다.

```text
result/
├── 00_결과요약.csv
├── 01_전체고객_위험순위.csv
├── 02_추가심사대상.csv
└── 03_1차기준통과.csv
```

## 판정 기준

- `RiskScore >= 0.074241900037`: `대출 보류(추가 심사)`
- `RiskScore < 0.074241900037`: `1차 기준 통과`

임계값은 개발 데이터의 5-fold OOF 예측에서 Recall 0.75 이상을 만족하는 후보 중 Precision이 가장 높은 값입니다. 법적 대출 승인·거절 기준이나 정확한 7.42% 연체확률을 뜻하지 않습니다.

## 가장 간단한 실행

1. 예측할 CSV를 `input/customer_data.csv`로 저장합니다.
2. `run_prediction.bat`을 실행합니다.
3. `result` 폴더의 CSV를 Excel에서 확인합니다.

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

Precision이 약 22.9%이므로 `추가 심사 대상`을 실제 대출 거절 고객으로 확정하면 안 됩니다. 이 프로그램의 결과는 추가 심사 순서를 정하기 위한 모델 위험 점수입니다.

## 모델 재학습

학습 데이터를 변경한 경우에만 실행합니다.

```powershell
python train_and_export_model.py --data-dir "학습데이터폴더경로"
```

일반적인 예측 실행에는 재학습이 필요하지 않습니다.
