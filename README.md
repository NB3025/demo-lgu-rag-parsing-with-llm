# 🚗 RAG with LLM - 자동차 매뉴얼 AI 어시스턴트

Amazon Bedrock, OpenSearch Serverless, S3를 활용한 한국어 자동차 매뉴얼 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 🎯 주요 기능

- **멀티모달 문서 처리**: PDF를 이미지로 변환하여 Claude 3.7 Sonnet으로 파싱
- **하이브리드 검색**: 벡터 검색 + 렉시컬 검색 결합
- **한국어 최적화**: Nori 한국어 분석기 적용
- **이미지 참조**: 각 페이지별 이미지를 S3에 저장하고 검색 결과에 표시
- **API 호출 제한 관리**: Claude 3.7 Sonnet의 1분당 5회 제한 자동 관리
- **스트리밍 응답**: 실시간 AI 응답 생성
- **웹 인터페이스**: Streamlit 기반 사용자 친화적 UI

## 🏗️ 시스템 아키텍처

```
📄 PDF 문서
    ↓ (PyMuPDF로 페이지별 이미지 변환)
🖼️ 페이지 이미지 (base64)
    ↓ (S3 저장)
☁️ S3 이미지 저장소
    ↓ (Claude 3.7 Sonnet 파싱)
📝 구조화된 텍스트
    ↓ (Titan2 임베딩)
🧠 벡터 임베딩
    ↓ (OpenSearch Serverless 인덱싱)
🔍 검색 가능한 지식베이스
    ↓ (하이브리드 검색)
📊 텍스트 + 이미지 참조 결과
```

## 🛠️ 기술 스택

- **LLM**: Claude 3.7 Sonnet (문서 파싱 및 응답 생성)
- **임베딩**: Amazon Titan Embeddings v2 (1024차원)
- **벡터 DB**: OpenSearch Serverless
- **객체 저장소**: Amazon S3
- **문서 처리**: PyMuPDF
- **웹 프레임워크**: Streamlit
- **언어 분석**: Nori Korean Analyzer

## 📋 사전 요구사항

### AWS 설정
```bash
# AWS CLI 설치 및 설정
aws configure
# 또는
aws configure --profile your-profile-name
```

### 필요한 AWS 권한
- Amazon Bedrock (Claude 3.7 Sonnet, Titan Embeddings v2)
- OpenSearch Serverless
- S3 (버킷 생성 및 객체 업로드)
- IAM (정책 생성)

### Python 환경
```bash
python >= 3.9
```

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd rag-with-llm
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 단계별 실행

#### Step 1: OpenSearch Serverless 컬렉션 생성
```bash
python step1_create_opensearch_collection.py
```
- 암호화, 네트워크, 데이터 액세스 정책 생성
- OpenSearch Serverless 컬렉션 생성
- OpenSearch 대시보드 액세스 활성화

#### Step 2: 벡터 인덱스 생성
```bash
python step2_create_index.py
```
- Titan2 임베딩용 1024차원 벡터 인덱스 생성
- Nori 한국어 분석기 설정
- 하이브리드 검색을 위한 인덱스 최적화

#### Step 3: 문서 처리 및 인덱싱
```bash
python step3_document_processing.py
```
- PDF를 페이지별 이미지로 변환
- Claude 3.7 Sonnet으로 구조화된 텍스트 추출
- 페이지 이미지를 S3에 저장
- Titan2로 임베딩 생성 및 OpenSearch에 인덱싱
- API 호출 제한 자동 관리 (1분당 5회)

#### Step 4: 검색 기능 테스트
```bash
python step4_search_test.py
```
- 벡터 검색 및 렉시컬 검색 테스트
- 하이브리드 검색 결과 확인

#### Step 5: RAG 채팅 테스트
```bash
python step5_rag_chat.py
```
- 컨텍스트 기반 질의응답 테스트
- Claude 3.5 Sonnet 스트리밍 응답

### 5. 웹 인터페이스 실행
```bash
streamlit run streamlit_rag_app.py
```

## 📁 프로젝트 구조

```
rag-with-llm/
├── step1_create_opensearch_collection.py  # OpenSearch 컬렉션 생성
├── step2_create_index.py                  # 벡터 인덱스 생성
├── step3_document_processing.py           # 문서 처리 및 인덱싱
├── step4_search_test.py                   # 검색 기능 테스트
├── step5_rag_chat.py                      # RAG 채팅 테스트
├── streamlit_rag_app.py                   # 웹 인터페이스
├── refer_parser_prompt.md                 # Claude 파싱 프롬프트
├── optimized-index-with-nori.json         # 인덱스 설정 템플릿
├── requirements.txt                       # Python 의존성
├── data/                                  # PDF 문서 저장 폴더
├── README.md                              # 프로젝트 문서
└── .gitignore                             # Git 제외 파일 목록
```

## 🔧 설정 파일

### opensearch_config.json (자동 생성)
```json
{
  "collection_name": "rag-car-manual-xxxxxx",
  "endpoint": "https://xxxxx.us-west-2.aoss.amazonaws.com",
  "region": "us-west-2",
  "policies": {
    "encryption": "policy-name",
    "network": "policy-name",
    "access": "policy-name"
  }
}
```

## 🎨 주요 특징

### 1. 페이지별 고유 URI
```
s3://bucket/documents/file.pdf#page=1
s3://bucket/documents/file.pdf#page=2
```

### 2. 이미지 참조 시스템
```
텍스트: s3://bucket/documents/file.pdf#page=1
이미지: s3://bucket/images/file_page_1.png
```

### 3. 하이브리드 검색
- **벡터 검색**: 의미적 유사성 기반
- **렉시컬 검색**: 키워드 매칭 기반
- **결합 검색**: 두 방식의 장점 결합

### 4. API 호출 제한 관리
```python
# Claude 3.7 Sonnet: 1분당 5회 제한
# 자동 대기 및 진행률 표시
⏳ API 호출 제한 도달 (5/5)
⏰ 45.2초 대기 중...
   ⏱️  44초 남음...
```

## 🔍 사용 예시

### 검색 쿼리
```
"차량 안전 기능에 대해 알려주세요"
"엔진 오일 교환 주기는 얼마나 되나요?"
"타이어 공기압 점검 방법"
```

### 응답 형태
- **텍스트 답변**: 관련 정보 요약
- **소스 참조**: 정확한 페이지 위치
- **이미지 표시**: 해당 페이지의 원본 이미지

## 🛡️ 보안 고려사항

- AWS 자격 증명은 환경 변수 또는 AWS CLI 프로필 사용
- 민감한 설정 파일은 `.gitignore`에 포함
- OpenSearch 네트워크 정책으로 액세스 제어
- S3 버킷 암호화 적용

## 📊 성능 최적화

- **벡터 차원**: 1024차원 (Titan2 최적화)
- **거리 메트릭**: L2 (OpenSearch Serverless 호환)
- **한국어 분석**: Nori 분석기 적용
- **청킹 전략**: 의미 단위 분할
- **캐싱**: S3 presigned URL 활용

## 🐛 문제 해결

### 일반적인 오류
1. **AWS 권한 오류**: IAM 권한 확인
2. **OpenSearch 연결 오류**: 네트워크 정책 확인
3. **Claude API 제한**: 자동 대기 기능 활용
4. **메모리 부족**: 문서 크기 및 배치 크기 조정

### 로그 확인
```bash
# 실행 로그 확인
tail -f nohup.out

# OpenSearch 대시보드 접속
https://your-collection-id.us-west-2.aoss.amazonaws.com/_dashboards/
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 Issue를 생성해 주세요.

---

**⚠️ 주의사항**: 
- 이 프로젝트는 AWS 서비스를 사용하므로 비용이 발생할 수 있습니다.
- PDF 문서는 저작권을 확인한 후 사용하세요.
- 프로덕션 환경에서는 추가적인 보안 설정이 필요합니다.
