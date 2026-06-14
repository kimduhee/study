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
