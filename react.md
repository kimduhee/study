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
 ├─ package-lock.json
 ├─ tsconfig.app.json
 ├─ tsconfig.json
 ├─ tsconfig.node.json
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
> + 토글
><pre><code>const [isOpen, setIsOpen] = useState(false);
>
>&lt;button onClick={() => setIsOpen(!isOpen)}&gt;열기&lt;/button&gt;
> 모달, 드롭다운
></code></pre>
> + 객체상태
><pre><code>const [user, setUser] = useState({ name: '', age: 0 });
>
>setUser({ ...user, name: '홍길동' });
></code></pre>

### useEffect (생명주기)
렌더링 이후에 실행해야 하는 로직을 처리
<pre><code>기본구조
useEffect(() => {
    // 실행할 코드 (effect)

    return () => {
        // cleanup (정리 작업)
    };
}, [dependency]);
</code></pre>

> 실행 타이밍
> + 처음 렌더링 후 1번만 실행
><pre><code>useEffect(() => {
>    console.log("처음 렌더링");
>}, []);
></code></pre>
> + 특정 값이 바뀔 때 실행
><pre><code>useEffect(() => {
>    console.log("count값 변경");
>}, [count]);
></code></pre>
> + 렌더링마다 실행
><pre><code>useEffect(() => {
>    console.log("렌더링마다 실행");
>});
></code></pre>

> cleanup 함수
> + 컴포넌트가 unmount 될 때, effect가 다시 실행되기 전 
><pre><code>useEffect(() => {
>    const id = setInterval(() => {
>        console.log("1초마다 실행");
>    }, 1000);
>
>    return () => {
>        clearInterval(id);
>    };
>}, []);
></code></pre>

> 자주 쓰는 패턴
> + API 호출
><pre><code>useEffect(() => {
>    fetch("/api.data")
>        .then(res => res.json())
>        .then(data => setDate(date));
>}, []);
>컴포넌트 로딩 시 데이터 가져오기
></code></pre>
> + props 변경 감지
><pre><code>useEffect(() => {
>    console.log("props 변경됨");
>}, [props.value]);
></code></pre>
> + 상태 동기화
><pre><code>useEffect(() => {
>    setFilteredList(list.filter(item => item.active));
>}, [list]
></code></pre>

### useRef (값 유지, DOM 접근)
값을 유지하거나 DOM 요소에 직접 접근할 때 사용하는 HOOK
<pre><code>기본구조
const ref = useRef(initialValue);
</code></pre>

> + DOM 요소 접근
><pre><code>import { useRef, useEffect } from "react";
>
>funtion InputFocus() {
>    const inputRef = useRef(null);
>
>    useEffect(() => {
>        inputRef.current.focus();
>    }, []);
>    
>    return <input ref={inputRef} />;
>}
></code></pre>
> + 값 저장(리렌더링 없이 유지)
><pre><code>import { useRef } from "react";
>
>function Counter() {
>    const countRef = useRef(0);
>
>    const increase = () => {
>        countRef.current += 1;
>        console.log(countRef.current);
>    };
>    return <button onClick={increase}>증가<button>;
>}
></code></pre>
> + 이전 값 저장
><pre><code>import { useRef, useEffect, useState } from "react";
>
>function PreviousValue() {
>    const [count, setCount] = useState(0);
>    const preCount = useRef(0);
>
>    useEffect(() => {
>        preCount.current = count;
>    }, [count]);
>
>    return (
>        <div>
>            <p>현재: { count }</p>
>            <p>이전: { preCount.current }</p>
>            <button onClick={() => setCount(count + 1)}>+</button>
>        </div>
>    );
>}
></code></pre>

### useMemo (최적화)
연산결과 캐싱

### useContext

### Zustand
전역 상태 관리(Global State Management) 도구

### 기타
<pre><code>(ref.file_name ?? "").trim();

=> ref.file_name이 null 또는 undefined이면 "" (빈 문자열)을 사용
</code></pre>
