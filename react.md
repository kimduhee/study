# 환경 구성
### Node.js 설치
<pre><code>node -v
npm -v
</code></pre>

### Vite 프로젝트 생성
<pre><code>npm create vite@latest
</code></pre>
> -프로젝트 이름: 본인의 프로젝트명(rag-system)<br>
> -프레임워크 선택: React<br>
> -Variant 선택: TypeScript

### 프로젝트 이동 및 설치
<pre><code>cd rag-system
npm install
</code></pre>

### 개발서버 실행
<pre><code>npm run dev
</code></pre>
> http://localhost:5173

### 기본구조
<pre><code>
my-app/
 ├─ node_modules/
 ├─ public/
 ├─ src/
 │   ├─ App.tsx
 │   ├─ main.tsx
 ├─ index.html
 ├─ package.json
 ├─ package-lock.json
 ├─ tsconfig.app.json
 ├─ tsconfig.json
 ├─ tsconfig.node.json
 └─ vite.config.js
</code></pre>
>node_modules<br/>
>+ npm으로 설치한 라이브러리 저장
>+ 예: react, axios, zustand 등

>public<br/>
>+ 정적 파일 (빌드 시 그대로 복사)
>+ 예:favicon.ico, robots.txt, 외부 이미지
>+ import 없이 /파일명으로 접근 가능
>+ React 코드에서 직접 import 안 해도 됨

>src<br/>
>+ 코드영역

>main.tsx<br/>
>+ 앱 시작점 (entry point)
>+ React 앱을 실제 DOM에 붙이는 역할
>+ index.html의 #root에 App을 렌더링

>App.tsx<br/>
>+ 최상위 컴포넌트
>+ 모든 화면의 시작점
>+ 라우팅
>+ 레이아웃
>+ 전역 상태 연결

>index.html<br/>
>+ 실제 웹 페이지의 기본 HTML
>+ React는 여기 안에 들어감
>+ #root가 React 앱의 시작 위치

>package.json<br/>
>+ 프로젝트 설정 + 라이브러리 목록
>+ 실행 명령어 정의 (npm run dev)
>+ 라이브러리 버전 관리

>package-lock.json<br/>
>+ 정확한 버전 고정 파일
>+ 팀원 간 동일 환경 보장
>+ “react 18.2.0 정확히 이 버전 써라” 기록

>tsconfig.json / tsconfig.*.json<br/>
>+ tsconfig.json: 공통 설정
>+ tsconfig.app.json: 프론트 코드용 설정 (src 기준)
>+ tsconfig.node.json: Node 환경 (vite.config 등) 설정

>vite.config.js<br/>
>+ Vite 설정 파일
>+ React 플러그인 적용
>+ alias 설정 (@/components)
>+ proxy 설정 (API 연결)
>+ 빌드 옵션



# 문법
### jsx 문법
JavaScript 안에서 HTML처럼 UI를 작성할 수 있게 해주는 확장 문법

##### JSX란
JS + HTML을 섞어서 쓰는 문법
<pre><code>const element = &lt;h1>Hello, world!&lt;/h1>;
</code></pre>
> 이 코드는 실제 내부적으로 다음과 같이 변환
<pre><code>const element = React.createElement("h1", null, "Hello, world!");
</code></pre>

##### JSX의 핵심 문법
JS + HTML을 섞어서 쓰는 문법

+ JavaScript 표현식 사용
<pre><code>const name = "Tom";
&lt;h1>Hello, {name}&lt;/h1>
</code></pre>

+ 속성(props) 사용
<pre><code>&lt;div className="box">&lt;/div>
</code></pre>
> class => className

+ 반드시 하나의 부모 요소
<pre><code>&lt;h1>Hello&lt;/h1>
&lt;p>World&lt;/p>
=> 오류

&lt;div>
  &lt;h1>Hello&lt;/h1>
  &lt;p>World&lt;/p>
&lt;/div>
=> 정상

&lt;>
  &lt;h1>Hello&lt;/h1>
  &lt;p>World&lt;/p>
&lt;/>
=> 정상
</code></pre>

+ 조건부 렌더링
<pre><code>const isLoggedIn = true;

{isLoggedIn ? &lt;p>환영합니다&lt;/p> : &lt;p>로그인하세요&lt;/p>}
</code></pre>

+ 리스트 렌더링
<pre><code>const items = ["A", "B", "C"];

&lt;ul>
  {items.map(item => (
    &lt;li key={item}>{item}&lt;/li>
  ))}
&lt;/ul>
</code></pre>

+ 이벤트 처리
<pre><code>&lt;button onClick={() => alert("클릭!")}>
  클릭
&lt;/button>
</code></pre>

### useState (상태)
React에서 컴포넌트 내부 상태(state)를 관리하는 가장 기본적인 Hook이며
화면에 표시되는 값을 기억하고, 바뀌면 자동으로 다시 렌더링해주는 기능이라고 보면 된다
+ 기본구조
<pre><code>import { useState } from 'react';

const Counter = () => {
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
</code></pre>
> count: 현재값<br>
> setCount: count 값 변경하는 함수<br>
> useState(0): 초기 렌더링 시 값(0)<br>
> 1. 처음 렌더링 시 count = 0<br>
> 2. 버튼 클릭 시 count + 1 한 값이 setCount에 의해 count에 저장<br>
> 3. 컴포넌트 다시 렌더링<br>
> 4. 화면 업데이트

+ 입력값
<pre><code>const [text, setText] = useState('');

&lt;input 
     value={text} 
     onChange={(e) => 
         setText(e.target.value)
     } 
/&gt; 
</code></pre>
> 로그인, 검색창, 채팅 입력창 필수 패턴

+ 토글
<pre><code>const [isOpen, setIsOpen] = useState(false);

&lt;button onClick={() => setIsOpen(!isOpen)}&gt;열기&lt;/button&gt;
</code></pre>
> 모달, 드롭다운

+ 객체상태
<pre><code>const [user, setUser] = useState({ name: '', age: 0 });

setUser({ ...user, name: '홍길동' });
</code></pre>

### useEffect (생명주기)
렌더링 이후에 실행해야 하는 로직을 처리
+ 기본구조
<pre><code>useEffect(() => {
    // 실행할 코드 (effect)

    return () => {
        // cleanup (정리 작업)
    };
}, [dependency]);
</code></pre>

+ 처음 렌더링 후 1번만 실행
<pre><code>useEffect(() => {
    console.log("처음 렌더링");
}, []);
</code></pre>

+ 특정 값이 바뀔 때 실행
<pre><code>useEffect(() => {
    console.log("count값 변경");
}, [count]);
</code></pre>

 + 렌더링마다 실행
<pre><code>useEffect(() => {
    console.log("렌더링마다 실행");
});
</code></pre>

+ cleanup(컴포넌트가 unmount 될 때, effect가 다시 실행되기 전)
<pre><code>useEffect(() => {
    const id = setInterval(() => {
        console.log("1초마다 실행");
    }, 1000);

    return () => {
        clearInterval(id);
    };
}, []);
</code></pre>

 + API 호출
<pre><code>useEffect(() => {
    fetch("/api.data")
        .then(res => res.json())
        .then(data => setDate(date));
}, []);
</code></pre>
> 컴포넌트 로딩 시 데이터 가져오기

+ props 변경 감지
<pre><code>useEffect(() => {
    console.log("props 변경됨");
}, [props.value]);
</code></pre>

+ 상태 동기화
<pre><code>useEffect(() => {
    setFilteredList(list.filter(item => item.active));
}, [list]
</code></pre>

### useRef (값 유지, DOM 접근)
값을 유지하거나 DOM 요소에 직접 접근할 때 사용하는 HOOK
+ 기본구조
<pre><code>const ref = useRef(initialValue);
</code></pre>

+ DOM 요소 접근
<pre><code>import { useRef, useEffect } from "react";

funtion InputFocus() {
    const inputRef = useRef(null);

    useEffect(() => {
        inputRef.current.focus();
    }, []);
    
    return &lt;input ref={inputRef} />;
}
</code></pre>

+ 값 저장(리렌더링 없이 유지)
<pre><code>import { useRef } from "react";

function Counter() {
    const countRef = useRef(0);

    const increase = () => {
        countRef.current += 1;
        console.log(countRef.current);
    };
    return &lt;button onClick={increase}>증가&lt;button>;
}
</code></pre>

+ 이전 값 저장
<pre><code>import { useRef, useEffect, useState } from "react";

function PreviousValue() {
    const [count, setCount] = useState(0);
    const preCount = useRef(0);

    useEffect(() => {
        preCount.current = count;
    }, [count]);

    return (
        &lt;div>
            &lt;p>현재: { count }&lt;/p>
            &lt;p>이전: { preCount.current }&lt;/p>
            &lt;button onClick={() => setCount(count + 1)}>+&lt;/button>
        &lt;/div>
    );
}
</code></pre>

### useMemo (최적화)
연산 결과를 메모이제이션(캐싱)해서 불필요한 재계산을 막는 HOOK
+ 기본구조
<pre><code>const memoizedValue = useMemo(() => {
    return 계산식;
}, [의존성]);
</code></pre>

+ 기본 사용
<pre><code>import { useMemo, useState } from "react";

function ExpensiveComponent() {
    const [count, setCount] = useState(0);
    const [text, setText] = useState("");

    const expensiveValue = useMemo(() => {
        console.log("계산 실행!");
        let result = 0;
        for (let i = 0; i < 10000000; i++) {
            result += i;
        }
        return result;
    }, [count]);

    return (
        &lt;div>
            &lt;p>값: {expensiveValue}&lt;/p>
            &lt;button onClick={() => setCount(count + 1)}count 증가&lt;/button>
            &lt;input value={text} onChange={(e) => setText(e.target.value)} />
        &lt;/div>
    );
}
</code></pre>
> - text만 바뀌면 계산 다시 안함<br>
> - count가 바뀔 때만 재계산

+ 객체/배열 재생성 방지
<pre><code>const user = useMemo(() => {
    return { name: "kim", age: 25 };
}, []);
</code></pre>

### useContext
React에서 전역 상태를 간단하게 공유하기 위한 Hook이며 
props를 계속 내려주는 “props drilling” 문제를 해결하는 데 핵심적

+ Context 생성
<pre><code>import { createContext } from "react";

export const ThemeContext = createContext();
</code></pre>

+ Provider로 감싸기
<pre><code>import { ThemeContext } from "./ThemeContext";

function App() {
  const theme = "dark";

  return (
    &lt;ThemeContext.Provider value={theme}>
      &lt;Child />
    &lt;/ThemeContext.Provider>
  );
}
</code></pre>

+ useContext로 사용
<pre><code>import { useContext } from "react";
import { ThemeContext } from "./ThemeContext";

function Child() {
  const theme = useContext(ThemeContext);

  return &lt;div>{theme}&lt;/div>;
}
</code></pre>

### useCallback
React에서 함수 재생성을 방지하기 위한 Hook
+ 기본구조
<pre><code>const memoizedFn = useCallback(() => {
  // 실행 로직
}, [dependencies]);
</code></pre>

+ useCallback 미사용
<pre><code>const Parent = () => {
  const [count, setCount] = useState(0);

  const handleClick = () => {
    console.log("클릭");
  };

  return &lt;Child onClick={handleClick} />;
};
</code></pre>
> + count 바뀔 때마다 handleClick 새로생성, Child 리렌더

+ useCallback 사용
<pre><code>const Parent = () => {
  const [count, setCount] = useState(0);

  const handleClick = useCallback(() => {
    console.log("클릭");
  }, []);

  return &lt;Child onClick={handleClick} />;
};
</code></pre>
> + 함수 재사용됨, Child 불필요한 리렌더 방지

### Redux
컴포넌트 간 상태 공유를 위한 부분으로 중앙 상태 관리 시스템이다. 특히 컴포넌트 트리가 깊어질수롤 prop 전달이 복잡해질 때 사용한다.

+ 기존 문제점
<pre><code>function App() {
  const [count, setCount] = useState(0);
  return &lt;Child count={count} setCount={setCount} />;
}
</code></pre>
> + 컴포넌트가 깊어질수록 props 계속 내려줘야 함(props drilling)
> + 여러 컴포넌트에서 같은 상태를 써야 하면 관리가 어려움

##### React + Redux 기본 구성
+ store 생성
<pre><code>// store.js
import { configureStore } from "@reduxjs/toolkit";
import counterReducer from "./counterSlice";

export const store = configureStore({
  reducer: {
    counter: counterReducer,
  },
});
</code></pre>

+ Provider로 React에 연결
<pre><code>import { Provider } from "react-redux";
import { store } from "./store";

function App() {
  return (
    &lt;Provider store={store}>
      &lt;Counter />
    &lt;/Provider>
  );
}
</code></pre>

+ slice 만들기 (Reducer + Action 합친 개념)
<pre><code>// counterSlice.js
import { createSlice } from "@reduxjs/toolkit";

const counterSlice = createSlice({
  name: "counter",
  initialState: { value: 0 },
  reducers: {
    increment: (state) => {
      state.value += 1; // 직접 수정처럼 보이지만 내부적으로 안전하게 처리됨
    },
    decrement: (state) => {
      state.value -= 1;
    },
  },
});

export const { increment, decrement } = counterSlice.actions;
export default counterSlice.reducer;
</code></pre>

+ React 컴포넌트에서 사용
<pre><code>import { useSelector, useDispatch } from "react-redux";
import { increment, decrement } from "./counterSlice";

function Counter() {
  const count = useSelector((state) => state.counter.value);
  const dispatch = useDispatch();

  return (
    &lt;div>
      &lt;h1>{count}&lt;/h1>
      &lt;button onClick={() => dispatch(increment())}>+&lt;/button>
      &lt;button onClick={() => dispatch(decrement())}>-&lt;/button>
    &lt;/div>
  );
}
</code></pre>

### Zustand
전역 상태 관리(Global State Management) 도구.

+ 설치
<pre><code>npm install zustand</code></pre>

+ store 생성
<pre><code>import { create } from 'zustand'

const useStore = create((set) => ({
  count: 0,
  increase: () => set((state) => ({ count: state.count + 1 })),
}))
</code></pre>

+ 컴포넌트에서 사용
<pre><code>function Counter() {
  const count = useStore((state) => state.count)
  const increase = useStore((state) => state.increase)

  return (
    &lt;div>
      &lt;p>{count}&lt;/p>
      &lt;button onClick={increase}>+&lt;/button>
    &lt;/div>
  )
}
</code></pre>

+ 상태값 localStorage에 저장
<pre><code>import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useStore = create(
  persist(
    (set) => ({
      count: 0,
      increase: () => set((s) => ({ count: s.count + 1 })),
    }),
    {
      name: 'counter-storage', // 저장 key
    }
  )
)
</code></pre>


### Router
페이지 새로고침 없이 URL만 바꿔서 화면을 전환하는 기술
+ 라우터 시작점
<pre><code>import { BrowserRouter } from "react-router-dom";

&lt;BrowserRouter>
  &lt;App />
&lt;/BrowserRouter>
</code></pre>

+ Routes,  Route
<pre><code>import { Routes, Route } from "react-router-dom";

&lt;Routes>
  &lt;Route path="/" element={&lt;Home />} />
  &lt;Route path="/about" element={&lt;About />} />
&lt;/Routes>
</code></pre>

+ 페이지 이동
<pre><code>import { useNavigate } from "react-router-dom";

const navigate = useNavigate();

navigate("/about");
</code></pre>

### 기타
+ null 또는 undefined이면 "" (빈 문자열)을 사용
<pre><code>(ref.file_name ?? "").trim();

=> ref.file_name이 null 또는 undefined이면 "" (빈 문자열)을 사용
</code></pre>

+ data 화면 처리
<pre><code>return (
    &lt;ul>
      {data.map((user) => (
        &lt;li key={user.id}>{user.name}&lt;/li>
      ))}
    &lt;/ul>
);
 
</code></pre>


