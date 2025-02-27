# 미국 ETF 큐레이션 서비스 기획

**기간**

**참여 인원**

**수상**

2024.09~2024.11 (3개월)

3명

2024년 NH투자증권 빅데이터 경진대회 입선상 수상

## 📌 개요

---

- **문제 정의**
    - 높은 수익성으로 미국 ETF 투자에 대한 수요, 관심 증가
    - 신규 고객 유치를 위해 미국 ETF 큐레이션 서비스 기획 필요
    - 기존 미국 시장 투자 고객들의 투자 성향 분석을 기반으로 고객 맞춤형 ETF 큐레이션 서비스 기획
- **결과물**
    - 미국 시장 투자 고객들의 투자 성향 분석 및 인사이트
    - 미국 ETF 큐레이션 서비스
- **분석 도구**
    - Python, Visual Studio, Tableau
- **역할**
    - 데이터 전처리 및 EDA
    - 군집 분석
    - 큐레이션 서비스 코드 설계
    - 발표 자료 제작 (PPT)

## 📌 분석 과정

### 1. 데이터 전처리

---

- **활용 데이터**
    - 미국 주식, ETF 관련 데이터 및 NH투자증권 고객의 투자 내역 데이터 활용 (NH투자증권측 제공)
    - 8개의 데이터셋을 분석에 활용: 주식 종목 정보, 종목 시세 정보, 고객 매수매도 정보, 고객 보유 정보, ETF 종목 정보, ETF 점수 정보
    - 종목와 고객의 거래 정보 관련 데이터셋의 경우 기간별 정보로 구성됨
- **전처리**
    - 분석 대상 선정: 2024.05.28~2024.08.26 동안의 정보가 모두 존재하는 종목 선정
    - 결측치 제거: 다수의 정보가 부재한 종목 제거
    - 파생변수 생성: 일일 변동폭 생성

### 2. EDA

---

- **상관관계**

![Image](https://github.com/user-attachments/assets/7b9a0a57-fd2b-4326-b9e4-8bcd02542baa)

: 거래량과 고객 조회, 관심 등록수 간의 상관관계가 높음 → 거래량이 많은 종목일 수록 고객의 관심이 높음

: 주식 가격과 수익/손실투자자비율 간의 상관관계가 높음 → 주식의 가격이 높을 수록 수익투자자비율이 낮아지고 손실투자자비율이 높아짐

### 3. 군집 분석

---

- **DTW 클러스터링을 활용한 미국 주식 군집 분석**
    - 주식은 시간에 따른 움직임이 중요할 것이라 예상해 시계열 특성을 고려해 유사성 계산 방법으로 DTW 기법 선택
    - 군집화를 위한 변수 전처리: 정규화 진행, 총보유수량은 다른 변수에 비해 단위 차이가 크기 때문에 정규화를 진행하기 전에 로그 변환 진행
    - 수익투자자비율, 일일변동폭, 총보유수량을 기준으로 군집화 진행

![Image](https://github.com/user-attachments/assets/3761e355-8ce0-4b30-8edf-053365dc14dd)

: WCSS - 군집 개수가 5개를 넘어가면 WCSS 감소 폭이 점점 작아지고, 군집 내 응집도를 크게 향상시키지 못함

: 실루엣 계수 - 군집 개수가 5개일 때 실루엣 계수가 비교적 높은 값을 가짐

: WCSS, 실루엣 계수를 고려해 군집 개수를 5개로 선정

: 군집 개수가 5개일 때 주식 종목이 균등하게 군집화됨 

- **미국 주식 군집 결과 해석**
    - 수익성, 안정성, 대중성을 기준으로 미국 주식 군집 특징을 파악
    
![Image](https://github.com/user-attachments/assets/e75e32d3-ab9c-4a0e-8d96-8e24476b122d)
    
![Image](https://github.com/user-attachments/assets/0e2b033f-9ad3-407c-8d5b-bb9ec68d67ab)


### 4. 고객군 투자 성향 분석

---

- **군집별 고객군 종목 구매 비율 비교**
    - 고객군을 고객 연령대, 투자 규모(금액), 투자 실력에 따라 분류
    - 고객군의 군집별 종목 구매 비율을 비교해 고객군의 투자 성향을 분석

![Image](https://github.com/user-attachments/assets/f7f69003-b94b-4778-b453-9a68edfc6aa0)


: 상관관계 결과를 고려했을 때, 주식의 가격이 종목 구매 비율의 차이에 영향을 미칠 수 있다고 유추 

### 5. 미국 ETF 큐레이션 서비스 구현

---

![Image](https://github.com/user-attachments/assets/9dbf7464-1d96-4a51-a4de-28f4162d5840)
- **미국 ETF 큐레이션 서비스 흐름 설계**
    - 고객 정보 입력 → 고객 투자 성향 계산/분석 → 추천 ETF 종목 선정 → 추천 ETF 종목 제시 → 궁금한 ETF 종목 검색 → ETF 종목 정보 제시 → 반복 혹은 종료
- **환경 설정 및 데이터 로드**
    - Microsoft Azure의 OpenAI와 Blob Storage를 활용하기 위해 환경 변수 로드, API 설정, 데이터 로드
- **사용자 투자 성향 분석**
    - 사용자와 유사한 고객군의 정보를 기반으로 투자 성향 분석
    - 사용자가 입력한 연령대, 투자 규모, 투자 실력을 기반으로 데이터 필터링
    - 군집 분석 결과를 기반으로 군집별 수익성, 안정성, 대중성 선호도 점수를 계산하여 사용자 투자 성향을 정량화
- **추천 ETF 종목 선정**
    - ETF의 수익률, 표준편차, 거래량 점수 데이터에 선호도 점수를 가중 평균하여 최적의 ETF 종목 선정
- **OpenAI를 활용한 ETF 특징 추출**
    - GPT 모델을 기반으로 프롬프트를 설계하고 조정하여 사용자에게 ETF의 특징을 반환
- **ETF 대시보드 제작**
    - 추후 사용자에게 ETF에 대한 상세 정보를 전달하기 위해 주가, 거래량, 변동률, 구성 종목, 섹터 정보 등을 포함한 대시보드 제작
- **UI 설계**
    - Gradio 패키지를 활용해 큐레이션 서비스를 UI로 구현 및 ETF 대시보드와 연결

## 📌 결론

---

- **미국 시장 투자 고객들의 투자 성향 분석 및 인사이트**
    - 고객군이 많이 구매하는 종목 군집을 기반으로 고객의 투자 성향 분석
    - 고객군에 따라 선호하는 투자 종목 특징이 다른 것을 확인
- **미국 ETF 큐레이션 서비스 구현**
    - 유사 고객들의 정보를 기반으로 투자 성향과 미국 ETF 종목 분석
    - 고객이 미국 ETF에 대한 정보를 손쉽게 수집할 수 있도록 큐레이션 서비스 구현
    - 나무증권 앱 내 미국 ETF 큐레이션 컨텐츠 활용 제안

## 📌 배운 점

---

- **데이터 주제 발굴 과정**
    - 데이터 스키마 및 관련 동향을 조사하여 분석 주제를 발굴하는 과정을 경험할 수 있었다.
- **대규모 데이터셋 분석 방법**
    - 분석 문제에 필요한 데이터와 변수를 선택하는 과정을 경험했다.
    - 날짜별 데이터를 다룰 때 어떤 값을 대푯값으로 사용해야 하는지를 배울 수 있었다.
    - 변수의 단위 차이가 존재할 때 정규화, 로그 변환 같은 변환 및 전처리를 통해 차이를 보정할 수 있음을 배웠다.
- **OpenAI 활용 방법**
    - OpenAI를 처음으로 활용해보면서 프롬프트가 결과에 미치는 영향을 체험할 수 있었다.
    - 비용 문제로 직접 학습까지 해볼 수는 없었지만 퓨샷러닝, 원샷러닝 등 데이터를 통해 AI 모델을 정교하게 학습하는 방법에 대해 배울 수 있었다.
