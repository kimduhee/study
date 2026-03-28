# 환경 구성
### Node.js 설치
<pre><code>node -v
npm -v
</code></pre>

### Vite 프로젝트 생성
<pre><code>npm create vite@latest
</code></pre>
+ 프로젝트 이름: 본인의 프로젝트명(rag-system)
+ 프레임워크 선택: React
+ Variant 선택: JavaScript 또는 TypeScript (추천: TS)

### 프로젝트 이동 및 설치
<pre><code>cd rag-system
npm install
</code></pre>

### 개발서버 실행
<pre><code>npm run dev
</code></pre>
+ http://localhost:5173

### 기본구조
<pre><code>
my-app/
 ├─ node_modules/
 ├─ public/
 ├─ src/
 │   ├─ App.jsx
 │   ├─ main.jsx
 ├─ index.html
 ├─ package.json
 └─ vite.config.js
</code></pre>


# 문법
<pre><code>(ref.file_name ?? "").trim();

=> ref.file_name이 null 또는 undefined이면 "" (빈 문자열)을 사용
</code></pre>
