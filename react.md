# React

## 1. 환경 구성

### 프로젝트 생성 (Vite)

```bash
npm create vite@latest
# 프로젝트명 입력 → React → TypeScript 선택

cd <프로젝트명>
npm install
npm run dev   # → http://localhost:5173
```

### 프로젝트 구조

```
my-app/
├─ public/              # 정적 파일 (빌드 시 그대로 복사, /파일명으로 접근)
├─ src/
│   ├─ main.tsx         # 앱 진입점, index.html의 #root에 App 렌더링
│   └─ App.tsx          # 최상위 컴포넌트 (라우팅, 레이아웃, 전역 상태)
├─ index.html           # 기본 HTML, #root가 React 마운트 위치
├─ package.json         # 라이브러리 목록 및 실행 명령어 정의
├─ tsconfig.json        # TypeScript 공통 설정
├─ tsconfig.app.json    # 프론트 코드(src) 전용 설정
├─ tsconfig.node.json   # Node 환경(vite.config 등) 설정
└─ vite.config.ts       # Vite 설정 (플러그인, alias, proxy, 빌드)
```

---

## 2. JSX 문법

JavaScript 안에서 HTML처럼 UI를 작성하는 확장 문법. 빌드 시 `React.createElement()` 호출로 변환됩니다.

```jsx
// JSX
const element = <h1>Hello, world!</h1>;

// 변환 결과
const element = React.createElement("h1", null, "Hello, world!");
```

### 핵심 규칙

```jsx
// 1. JS 표현식은 {} 안에
const name = "Tom";
<h1>Hello, {name}</h1>

// 2. class → className
<div className="box"></div>

// 3. 반드시 하나의 루트 요소 (또는 Fragment 사용)
<>
  <h1>Hello</h1>
  <p>World</p>
</>

// 4. 조건부 렌더링
{isLoggedIn ? <p>환영합니다</p> : <p>로그인하세요</p>}

// 5. 리스트 렌더링 - key 필수
{items.map(item => <li key={item}>{item}</li>)}

// 6. 이벤트 처리
<button onClick={() => alert("클릭!")}>클릭</button>
```

---

## 3. Hooks

### useState - 상태 관리

컴포넌트 내부 상태를 관리하며, 값이 바뀌면 자동으로 리렌더링합니다.

```jsx
const [count, setCount] = useState(0);                    // 숫자
const [text, setText] = useState('');                      // 입력값
const [isOpen, setIsOpen] = useState(false);               // 토글
const [user, setUser] = useState({ name: '', age: 0 });    // 객체

// 객체 업데이트는 스프레드 연산자 사용
setUser({ ...user, name: '홍길동' });
```

```jsx
function Counter() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>{count}</p>
      <button onClick={() => setCount(count + 1)}>증가</button>
    </div>
  );
}
```

---

### useEffect - 사이드 이펙트

렌더링 이후 실행할 로직(API 호출, 구독, 타이머 등)을 처리합니다.

```jsx
useEffect(() => { ... });           // 매 렌더링마다 실행
useEffect(() => { ... }, []);       // 마운트 시 1회만 실행
useEffect(() => { ... }, [count]);  // count 변경 시마다 실행
```

```jsx
// API 호출
useEffect(() => {
  fetch("/api/data")
    .then(res => res.json())
    .then(data => setData(data));
}, []);

// 타이머 - cleanup으로 메모리 누수 방지
useEffect(() => {
  const id = setInterval(() => console.log("1초마다"), 1000);
  return () => clearInterval(id);
}, []);

// 상태 동기화
useEffect(() => {
  setFilteredList(list.filter(item => item.active));
}, [list]);
```

---

### useRef - DOM 접근 / 값 유지

리렌더링 없이 값을 유지하거나 DOM 요소에 직접 접근할 때 사용합니다.

```jsx
// DOM 접근 (마운트 시 input에 포커스)
function InputFocus() {
  const inputRef = useRef(null);
  useEffect(() => { inputRef.current.focus(); }, []);
  return <input ref={inputRef} />;
}

// 리렌더링 없이 값 유지
function Counter() {
  const countRef = useRef(0);
  const increase = () => {
    countRef.current += 1;
    console.log(countRef.current);
  };
  return <button onClick={increase}>증가</button>;
}

// 이전 값 저장
function PreviousValue() {
  const [count, setCount] = useState(0);
  const prevCount = useRef(0);

  useEffect(() => {
    prevCount.current = count;
  }, [count]);

  return (
    <div>
      <p>현재: {count}</p>
      <p>이전: {prevCount.current}</p>
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}
```

---

### useMemo - 연산 최적화

연산 결과를 캐싱해 불필요한 재계산을 방지합니다.

```jsx
// count가 바뀔 때만 재계산, text 변경 시엔 캐시 반환
const expensiveValue = useMemo(() => {
  let result = 0;
  for (let i = 0; i < 10_000_000; i++) result += i;
  return result;
}, [count]);

// 객체/배열 재생성 방지
const user = useMemo(() => ({ name: "kim", age: 25 }), []);
```

---

### useCallback - 함수 재생성 방지

```jsx
// 미사용: count 바뀔 때마다 handleClick 새로 생성 → Child 불필요한 리렌더
const handleClick = () => console.log("클릭");

// 사용: 함수 재사용 → Child 리렌더 방지
const handleClick = useCallback(() => console.log("클릭"), []);

return <Child onClick={handleClick} />;
```

---

### useContext - 전역 상태 공유

props drilling 없이 컴포넌트 트리 어디서든 상태를 공유합니다.

```jsx
// 1. Context 생성
export const ThemeContext = createContext();

// 2. Provider로 감싸기
function App() {
  return (
    <ThemeContext.Provider value="dark">
      <Child />
    </ThemeContext.Provider>
  );
}

// 3. 하위 컴포넌트에서 사용
function Child() {
  const theme = useContext(ThemeContext);
  return <div>{theme}</div>;
}
```

---

### useActionState - 폼 액션 상태 관리 (React 19+)

폼 액션과 비동기 작업의 상태를 관리합니다.

```jsx
const [state, formAction, isPending] = useActionState(fn, initialState);
// state: 현재 상태 | formAction: form의 action에 전달 | isPending: 처리 중 여부
```

```jsx
async function submitForm(prevState, formData) {
  const name = formData.get("name");
  if (!name) return { error: "이름을 입력하세요" };
  return { success: `안녕하세요, ${name}` };
}

export default function MyForm() {
  const [state, formAction, isPending] = useActionState(submitForm, {});
  return (
    <form action={formAction}>
      <input name="name" />
      <button disabled={isPending}>제출</button>
      {state.error && <p>{state.error}</p>}
      {state.success && <p>{state.success}</p>}
    </form>
  );
}
```

---

## 4. 상태 관리

### Redux Toolkit

컴포넌트 트리가 깊거나 여러 컴포넌트가 같은 상태를 공유할 때 사용합니다.

```jsx
// 1. slice 생성 (Reducer + Action 통합)
const counterSlice = createSlice({
  name: "counter",
  initialState: { value: 0 },
  reducers: {
    increment: (state) => { state.value += 1; },
    decrement: (state) => { state.value -= 1; },
  },
});
export const { increment, decrement } = counterSlice.actions;
export default counterSlice.reducer;

// 2. store 생성
export const store = configureStore({
  reducer: { counter: counterReducer },
});

// 3. Provider로 연결
<Provider store={store}><App /></Provider>

// 4. 컴포넌트에서 사용
const count = useSelector((state) => state.counter.value);
const dispatch = useDispatch();
<button onClick={() => dispatch(increment())}>+</button>
```

---

### Zustand

Redux보다 간결한 전역 상태 관리 라이브러리입니다.

```bash
npm install zustand
```

```jsx
// store 생성
const useStore = create((set) => ({
  count: 0,
  increase: () => set((state) => ({ count: state.count + 1 })),
}));

// 컴포넌트에서 사용
function Counter() {
  const count = useStore((state) => state.count);
  const increase = useStore((state) => state.increase);
  return <div><p>{count}</p><button onClick={increase}>+</button></div>;
}

// localStorage 영속화
const useStore = create(
  persist(
    (set) => ({
      count: 0,
      increase: () => set((s) => ({ count: s.count + 1 })),
    }),
    { name: "counter-storage" }
  )
);
```

---

## 5. 라우팅 (React Router)

페이지 새로고침 없이 URL만 바꿔 화면을 전환합니다.

```jsx
// 진입점
<BrowserRouter><App /></BrowserRouter>

// 라우트 정의
<Routes>
  <Route path="/" element={<Home />} />
  <Route path="/about" element={<About />} />
</Routes>

// 페이지 이동
const navigate = useNavigate();
navigate("/about");
```

---

## 6. 환경 변수

| 파일 | 설명 |
|------|------|
| `.env.local` | 로컬 개발 (git 제외) |
| `.env.development` | 개발 환경 |
| `.env.staging` | 검증 환경 |
| `.env.production` | 운영 환경 |

```bash
# Vite는 VITE_ 접두사 필수
VITE_API_URL=https://api.example.com
```

```tsx
const apiUrl = import.meta.env.VITE_API_URL as string;
```

### 빌드

```bash
npm run build             # 운영 빌드 (dist/ 폴더 생성)
npm run build:development # 개발 환경 빌드
npm run build:staging     # 검증 환경 빌드

npx serve dist            # 빌드 결과 로컬 실행 테스트
```

---

## 7. 주요 이벤트 패턴

```jsx
// onClick
<button onClick={handleClick}>클릭</button>

// onChange (입력)
<input value={value} onChange={(e) => setValue(e.target.value)} />

// onKeyDown
<input onKeyDown={(e) => { if (e.key === "Enter") console.log("엔터"); }} />

// 체크박스
<input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />

// 파일 업로드
<input type="file" onChange={(e) => console.log(e.target.files[0])} />

// 리스트 렌더링
<ul>{data.map((user) => <li key={user.id}>{user.name}</li>)}</ul>

// null/undefined 방어
(ref.file_name ?? "").trim();
```

### debounce (검색 입력 최적화)

```jsx
function useDebounce(value, delay) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]); // delay도 의존성에 포함

  return debounced;
}

// 사용
const [text, setText] = useState("");
const debouncedText = useDebounce(text, 500);
```

---

## 8. Next.js

### 개요

Next.js는 React 기반의 풀스택 웹 프레임워크입니다. React만으로는 클라이언트 렌더링(CSR)만 가능하지만, Next.js는 SSR·SSG·ISR을 추가로 지원하고 파일 시스템 기반 라우팅, API Routes, 이미지 최적화 등을 내장합니다.

| 렌더링 방식 | 설명 | 적합 상황 |
|------------|------|----------|
| SSR (Server-Side Rendering) | 요청마다 서버에서 HTML 생성 | 실시간 데이터, 개인화 페이지 |
| SSG (Static Site Generation) | 빌드 시 HTML 미리 생성 | 블로그, 문서, 마케팅 페이지 |
| ISR (Incremental Static Regeneration) | SSG + 일정 주기 재생성 | 반정적 데이터 (상품 목록 등) |
| CSR (Client-Side Rendering) | 브라우저에서 JS로 렌더링 | 사용자 인터랙션 중심 UI |

---

### 프로젝트 생성

```bash
npx create-next-app@latest
# 프로젝트명 입력
# TypeScript: Yes
# ESLint: Yes
# Tailwind CSS: Yes/No
# src/ 디렉토리: Yes/No
# App Router: Yes (권장)

cd <프로젝트명>
npm run dev   # → http://localhost:3000
```

---

### 프로젝트 구조 (App Router 기준)

```
my-app/
├─ public/                  # 정적 파일 (/파일명으로 접근)
├─ src/
│   └─ app/
│       ├─ layout.tsx       # 공통 레이아웃 (전체 페이지 공통 적용)
│       ├─ page.tsx         # / 루트 페이지
│       ├─ globals.css      # 전역 CSS
│       ├─ about/
│       │   └─ page.tsx     # /about 페이지
│       ├─ blog/
│       │   ├─ page.tsx     # /blog 페이지
│       │   └─ [id]/
│       │       └─ page.tsx # /blog/:id 동적 라우트
│       └─ api/
│           └─ hello/
│               └─ route.ts # /api/hello API Route
├─ next.config.ts           # Next.js 설정
├─ tsconfig.json
└─ package.json
```

---

### App Router vs Pages Router

Next.js 13부터 App Router가 도입되었으며, 현재는 App Router 사용이 권장됩니다.

| 구분 | App Router (`app/`) | Pages Router (`pages/`) |
|------|---------------------|------------------------|
| 도입 | Next.js 13+ (권장) | Next.js 초기부터 |
| 기본 컴포넌트 | 서버 컴포넌트 | 클라이언트 컴포넌트 |
| 레이아웃 | `layout.tsx` 중첩 | `_app.tsx` 단일 |
| 데이터 페칭 | `async/await` 직접 | `getServerSideProps` 등 |
| API Routes | `app/api/route.ts` | `pages/api/*.ts` |

---

### 서버 컴포넌트 vs 클라이언트 컴포넌트

App Router에서 모든 컴포넌트는 기본적으로 **서버 컴포넌트**입니다.

```tsx
// 서버 컴포넌트 (기본값) - 파일 상단에 지시어 없음
// DB 접근, API 호출 가능 / useState·useEffect 사용 불가
async function ProductList() {
  const products = await fetch("https://api.example.com/products").then(r => r.json());
  return (
    <ul>
      {products.map(p => <li key={p.id}>{p.name}</li>)}
    </ul>
  );
}
```

```tsx
// 클라이언트 컴포넌트 - 파일 최상단에 "use client" 필수
"use client";

import { useState } from "react";

export default function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}
```

| 기능 | 서버 컴포넌트 | 클라이언트 컴포넌트 |
|------|:---:|:---:|
| useState / useEffect | ✗ | ✓ |
| DB / 파일시스템 접근 | ✓ | ✗ |
| 번들 크기 영향 | 없음 | 있음 |
| 이벤트 핸들러 | ✗ | ✓ |
| 브라우저 API | ✗ | ✓ |

---

### 라우팅 (App Router)

파일 시스템 구조가 그대로 URL이 됩니다.

```
app/page.tsx              → /
app/about/page.tsx        → /about
app/blog/[id]/page.tsx    → /blog/:id       (동적 라우트)
app/shop/[...slug]/page.tsx → /shop/a/b/c  (catch-all)
app/(auth)/login/page.tsx → /login          ((그룹)은 URL에 미포함)
app/dashboard/layout.tsx  → /dashboard 하위 공통 레이아웃
```

```tsx
// 동적 라우트 파라미터 사용
export default async function BlogPost({ params }: { params: { id: string } }) {
  const post = await fetch(`/api/posts/${params.id}`).then(r => r.json());
  return <h1>{post.title}</h1>;
}

// 클라이언트 사이드 이동
"use client";
import { useRouter } from "next/navigation";

const router = useRouter();
router.push("/about");
router.back();

// Link 컴포넌트 (prefetch 자동 지원)
import Link from "next/link";
<Link href="/about">About</Link>
```

---

### 레이아웃

`layout.tsx`는 해당 경로와 하위 모든 페이지에 공통으로 적용됩니다.

```tsx
// app/layout.tsx - 전체 공통 레이아웃
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <header>공통 헤더</header>
        <main>{children}</main>
        <footer>공통 푸터</footer>
      </body>
    </html>
  );
}

// app/dashboard/layout.tsx - /dashboard 하위에만 적용
export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div>
      <aside>사이드바</aside>
      <section>{children}</section>
    </div>
  );
}
```

---

### 데이터 페칭

```tsx
// 서버 컴포넌트에서 직접 fetch (SSR)
async function Page() {
  const data = await fetch("https://api.example.com/data", {
    cache: "no-store",          // SSR: 매 요청마다 새 데이터
    // cache: "force-cache",    // SSG: 빌드 시 1회 (기본값)
    // next: { revalidate: 60 } // ISR: 60초마다 재생성
  }).then(r => r.json());

  return <div>{data.title}</div>;
}

// 클라이언트 컴포넌트에서 fetch (CSR)
"use client";
export default function Page() {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch("/api/data").then(r => r.json()).then(setData);
  }, []);
  return <div>{data?.title}</div>;
}
```

---

### API Routes

`app/api/` 경로에 `route.ts` 파일로 백엔드 API를 작성합니다.

```tsx
// app/api/users/route.ts
import { NextRequest, NextResponse } from "next/server";

// GET /api/users
export async function GET() {
  const users = [{ id: 1, name: "kim" }];
  return NextResponse.json(users);
}

// POST /api/users
export async function POST(request: NextRequest) {
  const body = await request.json();
  return NextResponse.json({ created: body }, { status: 201 });
}
```

```tsx
// app/api/users/[id]/route.ts
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  return NextResponse.json({ id: params.id });
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  return NextResponse.json({ deleted: params.id });
}
```

---

### 환경 변수

```bash
# .env.local
NEXT_PUBLIC_API_URL=https://api.example.com  # 브라우저에서 접근 가능
DB_PASSWORD=secret                            # 서버에서만 접근 가능
```

```tsx
// 클라이언트 (NEXT_PUBLIC_ 접두사 필수)
const apiUrl = process.env.NEXT_PUBLIC_API_URL;

// 서버 컴포넌트 / API Route (접두사 불필요)
const dbPassword = process.env.DB_PASSWORD;
```

> Vite는 `VITE_` 접두사, Next.js는 `NEXT_PUBLIC_` 접두사를 사용합니다.

---

### 빌드 및 배포

```bash
npm run build   # .next/ 폴더에 최적화된 빌드 생성
npm run start   # 프로덕션 서버 실행 (빌드 후)
npm run lint    # ESLint 검사
```

```ts
// next.config.ts - 주요 설정
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ["cdn.example.com"],  // 외부 이미지 도메인 허용
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://backend:8080/:path*",  // API 프록시
      },
    ];
  },
};

export default nextConfig;
```

---

## 9. 외부 JS 파일 적용

### 방법 1: next/script (Next.js 권장)

Next.js 환경에서 외부 스크립트를 로드할 때 사용합니다. `strategy`로 로드 시점을 제어합니다.

```tsx
import Script from "next/script";

// app/layout.tsx 또는 개별 page.tsx에 추가
export default function Layout({ children }) {
  return (
    <html>
      <body>
        {children}

        {/* beforeInteractive: HTML 파싱 전 로드 (폴리필 등) */}
        <Script src="/js/polyfill.js" strategy="beforeInteractive" />

        {/* afterInteractive: 페이지 인터랙티브 후 로드 (기본값, GA 등) */}
        <Script src="https://cdn.example.com/sdk.js" strategy="afterInteractive" />

        {/* lazyOnload: 유휴 시간에 로드 (채팅 위젯 등) */}
        <Script src="https://cdn.example.com/chat.js" strategy="lazyOnload" />

        {/* 로드 완료 후 콜백 */}
        <Script
          src="https://cdn.example.com/map.js"
          strategy="afterInteractive"
          onLoad={() => console.log("지도 스크립트 로드 완료")}
          onError={() => console.error("스크립트 로드 실패")}
        />
      </body>
    </html>
  );
}
```

---

### 방법 2: public 폴더에 JS 파일 배치

프로젝트 내 자체 JS 파일(외부 솔루션 파일 등)을 적용할 때 사용합니다.

```
my-app/
└─ public/
    └─ js/
        └─ solution.js     # /js/solution.js 로 접근
```

```tsx
// Next.js: next/script로 로드
<Script src="/js/solution.js" strategy="afterInteractive" />

// React(Vite): index.html에 직접 추가
// index.html
<script src="/js/solution.js" defer></script>
```

---

### 방법 3: useEffect로 동적 로드 (React 범용)

조건부로 스크립트를 로드하거나, 특정 컴포넌트 마운트 시점에만 로드할 때 사용합니다.

```tsx
"use client";
import { useEffect } from "react";

export default function MapComponent() {
  useEffect(() => {
    // 이미 로드된 경우 중복 로드 방지
    if (document.getElementById("map-script")) return;

    const script = document.createElement("script");
    script.id = "map-script";
    script.src = "https://cdn.example.com/map.js";
    script.async = true;
    script.onload = () => {
      // 스크립트 로드 후 초기화
      window.initMap();
    };
    document.body.appendChild(script);

    return () => {
      // 컴포넌트 언마운트 시 정리 (선택)
      document.getElementById("map-script")?.remove();
    };
  }, []);

  return <div id="map" style={{ width: "100%", height: "400px" }} />;
}
```

---

### 방법 4: 인라인 스크립트 실행

서드파티 초기화 코드를 인라인으로 삽입할 때 사용합니다.

```tsx
// Next.js - dangerouslySetInnerHTML 사용
import Script from "next/script";

<Script id="gtag-init" strategy="afterInteractive">
  {`
    window.dataLayer = window.dataLayer || [];
    function gtag(){ dataLayer.push(arguments); }
    gtag('js', new Date());
    gtag('config', 'G-XXXXXXXXXX');
  `}
</Script>
```

---

### 외부 JS에서 선언된 전역 변수 사용 (TypeScript)

외부 스크립트가 `window` 객체에 전역 변수를 추가하는 경우 타입 오류가 발생합니다.

```ts
// types/global.d.ts - 전역 타입 선언
declare global {
  interface Window {
    kakao: any;        // 카카오 지도 SDK
    gtag: Function;    // Google Analytics
    initMap: () => void;
  }
}

export {};
```

```tsx
// 사용
useEffect(() => {
  if (window.kakao) {
    window.kakao.maps.load(() => {
      const map = new window.kakao.maps.Map(/* ... */);
    });
  }
}, []);
```

---

### 방법 비교

| 방법 | 환경 | 로드 제어 | 적합 상황 |
|------|------|----------|----------|
| `next/script` | Next.js | strategy로 세밀하게 제어 | CDN 라이브러리, GA, 채팅 위젯 |
| `public/` + Script 태그 | React/Next.js | defer / async | 내부 솔루션 JS 파일 |
| `useEffect` 동적 로드 | React/Next.js | 마운트 시점 | 조건부·컴포넌트 단위 로드 |
| 인라인 스크립트 | React/Next.js | 즉시 실행 | 초기화 코드, 트래킹 픽셀 |

---

## 10. 빌드 난독화 (Build Obfuscation)

### 개념

| 구분 | 설명 | 적용 범위 |
|------|------|----------|
| **Minification (압축)** | 공백·주석 제거, 변수명 단축 | 기본 빌드에 포함 |
| **Obfuscation (난독화)** | 코드 구조 변형, 문자열 암호화, 제어 흐름 평탄화 | 별도 설정 필요 |

> 난독화는 완전한 보안 수단이 아니며 해독 시간을 높이는 것이 목적입니다.
> 난독화 수준이 높을수록 번들 크기 증가·실행 성능 저하가 발생합니다.

---

### React (Vite) 난독화

#### 기본 압축 설정 (esbuild / terser)

Vite는 기본적으로 esbuild로 압축합니다. 더 강력한 압축이 필요하면 terser를 사용합니다.

```bash
npm install --save-dev terser
```

```ts
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    minify: "terser",       // 기본값: "esbuild" (빠름) | "terser" (더 강력)
    terserOptions: {
      compress: {
        drop_console: true,   // console.log 제거
        drop_debugger: true,  // debugger 구문 제거
        pure_funcs: ["console.info", "console.warn"],
      },
      mangle: {
        toplevel: true,       // 최상위 변수·함수명 난독화
      },
      format: {
        comments: false,      // 주석 전체 제거
      },
    },
  },
});
```

#### javascript-obfuscator 플러그인 (고급 난독화)

```bash
npm install --save-dev vite-plugin-obfuscator
```

```ts
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import obfuscatorPlugin from "vite-plugin-obfuscator";

export default defineConfig({
  plugins: [
    react(),
    obfuscatorPlugin({
      apply: "build",           // 빌드 시에만 적용
      options: {
        // 문자열 배열로 추출 후 간접 참조
        stringArray: true,
        stringArrayEncoding: ["base64"],   // "none" | "base64" | "rc4"
        stringArrayThreshold: 0.75,        // 75% 확률로 문자열 추출

        // 제어 흐름 평탄화 (성능 저하 주의)
        controlFlowFlattening: true,
        controlFlowFlatteningThreshold: 0.5,

        // 변수·함수명 난독화
        identifierNamesGenerator: "hexadecimal",  // "hexadecimal" | "mangled"

        // 데드 코드 삽입
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 0.2,

        // 디버깅 방지
        disableConsoleOutput: true,
        debugProtection: false,            // true 설정 시 DevTools 성능 저하

        // 소스맵 제거
        sourceMap: false,
      },
    }),
  ],
});
```

---

### Next.js 난독화

#### SWC 기본 압축 (기본 활성화)

Next.js는 기본적으로 SWC로 압축합니다. 별도 설정 없이도 minification이 적용됩니다.

```ts
// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  swcMinify: true,   // 기본값 true, SWC 기반 압축
};

export default nextConfig;
```

#### Webpack + javascript-obfuscator (고급 난독화)

```bash
npm install --save-dev webpack-obfuscator javascript-obfuscator
```

```ts
// next.config.ts
import type { NextConfig } from "next";
import WebpackObfuscator from "webpack-obfuscator";

const nextConfig: NextConfig = {
  webpack(config, { dev, isServer }) {
    // 프로덕션 빌드, 클라이언트 번들에만 적용
    if (!dev && !isServer) {
      config.plugins.push(
        new WebpackObfuscator(
          {
            stringArray: true,
            stringArrayEncoding: ["base64"],
            stringArrayThreshold: 0.75,
            controlFlowFlattening: true,
            controlFlowFlatteningThreshold: 0.3,
            identifierNamesGenerator: "hexadecimal",
            deadCodeInjection: false,    // 번들 크기 증가 주의
            disableConsoleOutput: true,
            sourceMap: false,
          },
          ["excluded_bundle.js"]         // 난독화 제외할 파일
        )
      );
    }
    return config;
  },
};

export default nextConfig;
```

---

### 난독화 주요 옵션

| 옵션 | 설명 | 성능 영향 |
|------|------|----------|
| `stringArray` | 문자열을 배열로 추출 후 인덱스로 참조 | 낮음 |
| `stringArrayEncoding` | 추출된 문자열 인코딩 (`base64`, `rc4`) | 낮음 |
| `controlFlowFlattening` | if/else·loop를 switch 구조로 평탄화 | **높음** |
| `deadCodeInjection` | 실행되지 않는 가짜 코드 삽입 | 중간 |
| `identifierNamesGenerator` | 변수·함수명을 16진수 등으로 변환 | 낮음 |
| `disableConsoleOutput` | `console.*` 호출 제거 | 없음 |
| `debugProtection` | DevTools 열면 무한 루프 진입 | 없음 (UX 주의) |
| `sourceMap: false` | 소스맵 생성 비활성화 | 없음 |

---

### 난독화 수준별 권장 설정

```ts
// 수준 1 - 기본 (성능 영향 최소)
{
  stringArray: true,
  stringArrayEncoding: ["base64"],
  stringArrayThreshold: 0.5,
  identifierNamesGenerator: "hexadecimal",
  disableConsoleOutput: true,
  sourceMap: false,
}

// 수준 2 - 중간
{
  stringArray: true,
  stringArrayEncoding: ["base64"],
  stringArrayThreshold: 0.75,
  controlFlowFlattening: true,
  controlFlowFlatteningThreshold: 0.3,
  identifierNamesGenerator: "hexadecimal",
  deadCodeInjection: true,
  deadCodeInjectionThreshold: 0.2,
  disableConsoleOutput: true,
  sourceMap: false,
}

// 수준 3 - 강력 (번들 크기·성능 저하 감수)
{
  stringArray: true,
  stringArrayEncoding: ["rc4"],
  stringArrayThreshold: 1,
  controlFlowFlattening: true,
  controlFlowFlatteningThreshold: 0.75,
  identifierNamesGenerator: "hexadecimal",
  deadCodeInjection: true,
  deadCodeInjectionThreshold: 0.4,
  disableConsoleOutput: true,
  debugProtection: true,
  sourceMap: false,
}
```

---

### 소스맵 노출 방지

프로덕션 빌드에서 소스맵이 노출되면 난독화가 무의미해집니다.

```ts
// vite.config.ts
export default defineConfig({
  build: {
    sourcemap: false,           // 소스맵 비활성화 (기본값: false)
    // sourcemap: "hidden",     // 소스맵 생성하되 번들에 URL 미포함 (에러 추적용)
  },
});

// next.config.ts
const nextConfig: NextConfig = {
  productionBrowserSourceMaps: false,  // 기본값: false
};
```

| `sourcemap` 값 | 설명 |
|---------------|------|
| `false` | 소스맵 미생성 (프로덕션 권장) |
| `true` | 소스맵 생성 + 번들에 URL 포함 (노출 위험) |
| `"hidden"` | 소스맵 생성하되 번들에 URL 미포함 (Sentry 등 에러 추적용) |

---

## 11. 브라우저 스토리지

### 종류 비교

| 구분 | localStorage | sessionStorage | Cookie | IndexedDB |
|------|:---:|:---:|:---:|:---:|
| 만료 | 명시적 삭제 전까지 영구 | 탭/창 닫으면 삭제 | 만료일 설정 가능 | 명시적 삭제 전까지 영구 |
| 용량 | ~5~10MB | ~5~10MB | ~4KB | 수백MB 이상 |
| 서버 전송 | 안 됨 | 안 됨 | 매 요청마다 자동 전송 | 안 됨 |
| 저장 형식 | 문자열만 | 문자열만 | 문자열만 | 객체/Blob 등 다양 |
| 용도 | 로그인 토큰, 설정값 | 탭 단위 임시 상태 | 인증 세션(서버 검증 필요 시) | 대용량 구조화 데이터 |

> 민감한 인증 정보(JWT 등)는 XSS에 취약한 localStorage보다 `httpOnly` 쿠키 저장이 보안상 더 안전합니다.

---

### localStorage / sessionStorage 기본 사용

```js
// 저장 - 값은 반드시 문자열, 객체는 JSON.stringify 필요
localStorage.setItem("token", "abc123");
localStorage.setItem("user", JSON.stringify({ id: 1, name: "kim" }));

// 조회
const token = localStorage.getItem("token");
const user = JSON.parse(localStorage.getItem("user") ?? "null");

// 삭제
localStorage.removeItem("token");
localStorage.clear(); // 전체 삭제

// sessionStorage는 API 동일, 탭/창 닫으면 자동 삭제
sessionStorage.setItem("formDraft", JSON.stringify(formData));
```

---

### useState와 localStorage 동기화 (커스텀 훅)

```tsx
import { useState, useEffect } from "react";

function useLocalStorage<T>(key: string, initialValue: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue] as const;
}

// 사용
function Settings() {
  const [theme, setTheme] = useLocalStorage("theme", "light");
  return (
    <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
      현재 테마: {theme}
    </button>
  );
}
```

### 다른 탭 간 동기화 (storage 이벤트)

`storage` 이벤트는 **같은 키를 변경한 다른 탭**에서만 발생합니다 (현재 탭 자신은 감지 안 됨).

```tsx
useEffect(() => {
  const handleStorageChange = (e: StorageEvent) => {
    if (e.key === "token" && e.newValue === null) {
      // 다른 탭에서 로그아웃(토큰 삭제) 시 현재 탭도 동기화
      window.location.href = "/login";
    }
  };

  window.addEventListener("storage", handleStorageChange);
  return () => window.removeEventListener("storage", handleStorageChange);
}, []);
```

---

### Cookie 사용 (js-cookie)

서버로 자동 전송되어야 하는 값(세션 등)은 쿠키를 사용합니다.

```bash
npm install js-cookie
npm install --save-dev @types/js-cookie
```

```tsx
import Cookies from "js-cookie";

// 저장 (만료일, 옵션 설정 가능)
Cookies.set("refreshToken", "xyz789", {
  expires: 7,          // 7일 후 만료
  secure: true,         // HTTPS에서만 전송
  sameSite: "strict",   // CSRF 방지
});

// 조회
const token = Cookies.get("refreshToken");

// 삭제
Cookies.remove("refreshToken");
```

> `httpOnly` 쿠키는 보안상 JS로 직접 설정/조회가 불가능하며, 반드시 **서버 응답의 `Set-Cookie` 헤더**로만 설정 가능합니다. 인증 토큰은 가능하면 이 방식을 사용합니다.

```ts
// 서버(Express)에서 httpOnly 쿠키 설정 예시
res.cookie("accessToken", token, {
  httpOnly: true,   // JS에서 document.cookie로 접근 불가 (XSS 방어)
  secure: true,
  sameSite: "strict",
  maxAge: 1000 * 60 * 60, // 1시간
});
```

---

### IndexedDB (대용량 구조화 데이터)

복잡한 쿼리나 대용량 데이터가 필요하면 `idb` 같은 래퍼 라이브러리를 사용합니다.

```bash
npm install idb
```

```ts
import { openDB } from "idb";

async function getDb() {
  return openDB("my-app-db", 1, {
    upgrade(db) {
      db.createObjectStore("cache", { keyPath: "id" });
    },
  });
}

// 저장
async function saveCache(id: string, data: unknown) {
  const db = await getDb();
  await db.put("cache", { id, data, savedAt: Date.now() });
}

// 조회
async function getCache(id: string) {
  const db = await getDb();
  return db.get("cache", id);
}
```

---

### React Query / Zustand persist와의 비교

상태 관리 라이브러리에서 자동으로 스토리지 동기화를 처리해주는 경우도 많습니다.

```tsx
// Zustand persist - localStorage 자동 동기화 (앞선 "4. 상태 관리" 섹션 참고)
import { create } from "zustand";
import { persist } from "zustand/middleware";

const useAuthStore = create(
  persist(
    (set) => ({
      token: null,
      setToken: (token: string) => set({ token }),
      logout: () => set({ token: null }),
    }),
    { name: "auth-storage" } // localStorage에 "auth-storage" 키로 자동 저장/복원
  )
);
```

| 방법 | 적합 상황 |
|------|----------|
| 직접 `localStorage` API | 단순 값 1~2개, 빠른 구현 |
| `useLocalStorage` 커스텀 훅 | 컴포넌트 상태와 자동 동기화 필요 시 |
| `js-cookie` | 서버 전송이 필요한 값 |
| `httpOnly` 쿠키(서버 설정) | 인증 토큰 등 보안이 중요한 값 |
| `idb` (IndexedDB) | 대용량·구조화된 오프라인 데이터 |
| Zustand `persist` 등 | 전역 상태 + 영속화를 동시에 관리할 때 |
