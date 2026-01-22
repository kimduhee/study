1. Ollama 설치
https://ollama.com

설치확인 
ollama --version

모델다운로드
ollama pull llama3.1:8b

CPU: llama3.1:8b
GPU 있음: llama3.1:70b

실행
ollama run llama3.1:8b

엔드포인트
POST http://localhost:11434/api/generate
POST http://localhost:11434/api/embeddings
