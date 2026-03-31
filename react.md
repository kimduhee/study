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
+ Variant 선택: TypeScript

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
### useState (상태)
React에서 컴포넌트 내부 상태(state)를 관리하는 가장 기본적인 Hook이며
화면에 표시되는 값을 기억하고, 바뀌면 자동으로 다시 렌더링해주는 기능이라고 보면 된다
<pre><code>import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    &lt;div&gt;
      &lt;p&gt;{count}&lt;/p&gt;
      &lt;button 
          onClick={() => 
              setCount(count + 1)
          }&gt;
              증가
          &lt;/button&gt;
    &lt;/div&gt;
  );
}

1. 처음 렌더링 시 count = 0
2. 버튼 클릭 시 count + 1 한 값이 setCount에 의해 count에 저장
3. 컴포넌트 다시 렌더링
4. 화면 업데이트
</code></pre>
+ count: 현재값
+ setCount: count 값 변경하는 함수
+ useState(0): 초기 렌더링 시 값(0)

> 자주 쓰는 패턴
> + 입력값
><pre><code>const [text, setText] = useState('');
>
>&lt;input 
>     value={text} 
>     onChange={(e) => 
>         setText(e.target.value)
>     } 
>/&gt;
> 로그인, 검색창, 채팅 입력창 필수 패턴 
></code></pre>

### useEffect (생명주기)
Api 호출, 초기화, 이벤트 처리

### useRef (값 유지, DOM 접근)
렌더링과 관련없이 값 유지

### useMemo (최적화)
연산결과 캐싱

### useContext

### Zustand
전역 상태 관리(Global State Management) 도구

### 기타
<pre><code>(ref.file_name ?? "").trim();

=> ref.file_name이 null 또는 undefined이면 "" (빈 문자열)을 사용
</code></pre>
