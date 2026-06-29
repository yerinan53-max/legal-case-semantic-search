# Legal Case Finder

법률 문서 임베딩을 활용해 입력한 사건 설명과 의미가 유사한 판례를 검색하고,
검색 결과에서 공통 쟁점을 추출하는 딥러닝 수업 프로젝트입니다.

> 현재 `data/sample_cases.csv`는 기능 검증을 위한 교육용 합성 데이터입니다.
> 실제 분석 결과로 사용하지 마세요.

실제 학습 데이터는 AI Hub의 **법률/규정 텍스트 분석 데이터(고도화) -
상황에 따른 판례 데이터**를 신청하여 로컬에서만 사용합니다. 원본 및 가공
데이터는 이 저장소에 포함하지 않습니다. 자세한 준수 기준은
[`DATA_USAGE_POLICY.md`](DATA_USAGE_POLICY.md)를 확인하세요.

## 주요 기능

- 한국어 Sentence-BERT 기반 판례 임베딩
- 코사인 유사도를 이용한 유사 판례 Top-K 검색
- 검색 결과의 사건 분야 분포와 공통 쟁점 키워드 분석
- CSV 업로드 및 임베딩 인덱스 재생성
- 법률 데이터 미세조정과 검색 성능 평가용 Jupyter Notebook

## 프로젝트 구조

```text
PROJECT2/
├─ app.py
├─ data/
│  ├─ sample_cases.csv
│  └─ training_pairs.csv
├─ notebooks/
│  └─ train_embedding_model.ipynb
├─ scripts/
│  └─ build_index.py
├─ src/
│  ├─ analysis.py
│  ├─ config.py
│  ├─ data.py
│  ├─ embedding.py
│  └─ search.py
├─ tests/
│  └─ test_core.py
└─ requirements.txt
```

## 실행 방법

PowerShell에서 다음 명령을 실행합니다.

```powershell
cd C:\Python310\PROJECT2
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\build_index.py
streamlit run app.py
```

첫 실행 시 Hugging Face에서 약 443MB의 모델 가중치를 내려받습니다.

## 실제 데이터 형식

CSV 파일은 UTF-8로 저장하고 다음 열을 포함해야 합니다.

| 열 | 설명 |
|---|---|
| `id` | 고유 식별자 |
| `case_name` | 사건명 |
| `court` | 법원 |
| `decision_date` | 선고일 |
| `case_number` | 사건번호 |
| `category` | 사건 분야 |
| `issues` | 핵심 쟁점 |
| `summary` | 판결 요약 |
| `text` | 판례 본문 또는 판결 요지 |
| `source_url` | 출처 주소 |

국가법령정보 공동활용 Open API에서 판례 목록과 본문을 수집할 수 있습니다.
연구·수업 제출물에는 데이터 출처와 이용 조건을 반드시 명시하세요.

## 학습 및 평가

VS Code에서 `notebooks/train_embedding_model.ipynb`를 열고 커널을
`.venv\Scripts\python.exe`로 선택합니다. 노트북은 다음 과정을 포함합니다.

1. 데이터 탐색 및 전처리
2. 기본 Sentence-BERT 모델 임베딩 생성
3. Recall@K 기반 검색 성능 평가
4. 유사 판례 쌍을 이용한 선택적 미세조정
5. 학습 모델 저장과 전체 판례 인덱스 생성

## 주의사항

이 앱은 교육 및 연구용이며 법률 자문을 제공하지 않습니다. 검색 결과는 실제
판례 원문과 최신 법령을 통해 다시 확인해야 합니다.

이 프로젝트는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을
받아 구축된 AI Hub 법률 데이터를 학습 목적으로 활용합니다. 실제 데이터로
학습한 결과물을 공개하거나 제출할 때에는 AI Hub의 최신 의무 인용 문구와 해당
데이터셋의 개별 이용조건을 다시 확인해야 합니다.
