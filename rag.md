# 1. Ollama 설치
https://ollama.com
### 설치확인
<pre><code>ollama --version
</code></pre>
### 모델다운로드
CPU: llama3.1:8b
GPU 있음: llama3.1:70b
<pre><code>ollama pull llama3.1:8b
</code></pre>
### 실행
<pre><code>ollama run llama3.1:8b
</code></pre>
### 엔드포인트
<pre><code>POST http://localhost:11434/api/generate
POST http://localhost:11434/api/embeddings
</code></pre>


