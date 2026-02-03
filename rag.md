# 1. Ollama 설치
https://ollama.com
### 설치확인
<pre><code>ollama --version
</code></pre>
### 모델다운로드
+ CPU: llama3.1:8b
+ GPU 있음: llama3.1:70b
<pre><code>ollama pull llama3.1:8b
</code></pre>
### 실행
<pre><code>ollama run llama3.1:8b
</code></pre>
### 엔드포인트
<pre><code>POST http://localhost:11434/api/generate
POST http://localhost:11434/api/embeddings
</code></pre>
### 어플리케이션과 ollama가 별도 컴퓨터 구동시
ollama 설치된 서버에서 설정
+ 리눅스, macOS
<pre><code>bash
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
</code></pre>
+ window
<pre><code>powershell
setx OLLAMA_HOST "0.0.0.0:11434"
ollama serve
</code></pre>

# 99. 기타
### SentenceTransformer
문장을 숫자 벡터(embedding)로 바꿔주는 모델 래퍼이며 
문장의 의미를 좌표로 만든다
#### 1)MiniLM 계열(가볍고 빠름)
대표모델
+ all-MiniLM-L6-v2
+ all-MiniLM-L12-v2

특징
+ 빠름
+ 가벼움
+ 성능은 중간

<pre><code>SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
</code></pre>

| 항목 | 값 |
|:---|:---|
| 차원 | 384 |
| 언어 | 영어 위주 |
| 용도 | PoC, 테스트, 소규모 검색 |

#### 2)MPNet 계열(영어 검색 강자)
대표모델
+ all-mpnet-base-v2

특징
+ MiniLM보다 정확
+ 영어 의미 검색 성능 매우 좋음

<pre><code>SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
</code></pre>

| 항목 | 값 |
|:---|:---|
| 차원 | 768 |
| 언어 | 영어 |
| 용도 | 영어 문서 RAG |

#### 3)BGE 계열(요즘 실무 표준)
대표모델
+ BAAI/bge-small-en
+ BAAI/bge-base-en
+ BAAI/bge-large-en
+ BAAI/bge-m3(권장)

특징
+ 멀티언어(한국어 매우 좋음)
+ 검색/질의응답 특화 학습
+ Reranker까지 풀세트 

<pre><code>SentenceTransformer("BAAI/bge-m3")
</code></pre>

| 모델 | 차원 | 비고 |
|:---|:---|:---|
| bge-small | 384 | 가벼움 |
| bge-base | 768 | 균형 |
| bge-large | 1024 | 고성능 |
| bge-m3 | 1024 | 멀티언어 최강 |

#### 4)E5 계열 (Query/Passage 분리형)
대표모델
+ intfloat/e5-small
+ intfloat/e5-base
+ intfloat/e5-large

특징
+ 검색 성능 매우 우수
+ prefix 규칙 필수

<pre><code>SentenceTransformer("intfloat/e5-base")

query = "query: FAISS 역할"
doc   = "passage: FAISS는 벡터 DB이다"
</code></pre>

#### 5)GTE 계열 (Alibaba)
대표모델
+ thenlper/gte-base
+ thenlper/gte-large

특징
+ BGE와 비슷한 성향
+ 한국어도 준수
+ 상대적으로 덜 알려짐

#### 6)다국어 / 한국어 특화 모델
대표모델
+ jhgan/ko-sroberta-multitask
+ snunlp/KR-SBERT-V40K-klueNLI
+ sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

<pre><code>SentenceTransformer("jhgan/ko-sroberta-multitask")
</code></pre>


