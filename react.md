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
