# Vue.js

## 1. 환경 구성

### 프로젝트 생성 (Vite)

```bash
npm create vite@latest
# 프로젝트명 입력 → Vue → TypeScript 선택

cd <프로젝트명>
npm install
npm run dev   # → http://localhost:5173
```

### 프로젝트 구조

```
my-app/
├─ public/              # 정적 파일 (빌드 시 그대로 복사, /파일명으로 접근)
├─ src/
│   ├─ assets/          # 이미지, 폰트 등 빌드에 포함되는 리소스
│   ├─ components/      # 재사용 컴포넌트
│   ├─ App.vue          # 최상위 컴포넌트
│   └─ main.ts          # 앱 진입점
├─ index.html           # 기본 HTML, #app이 Vue 마운트 위치
├─ package.json
├─ tsconfig.json
└─ vite.config.ts
```

```ts
// main.ts - 앱 진입점
import { createApp } from "vue";
import App from "./App.vue";

createApp(App).mount("#app");
```

### SFC (Single File Component) 구조

Vue 컴포넌트는 `.vue` 파일 하나에 템플릿·로직·스타일을 모두 담는 SFC 방식을 사용합니다.

```vue
<template>
  <div class="box">{{ message }}</div>
</template>

<script setup lang="ts">
import { ref } from "vue";
const message = ref("Hello Vue");
</script>

<style scoped>
.box {
  color: blue;
}
</style>
```

| 블록 | 설명 |
|------|------|
| `<template>` | HTML 기반 마크업 |
| `<script setup>` | Composition API 로직 (컴파일 타임에 최적화) |
| `<style scoped>` | 해당 컴포넌트에만 적용되는 CSS |

---

## 2. 템플릿 문법

### 텍스트 바인딩 및 디렉티브

```vue
<template>
  <!-- 텍스트 보간 -->
  <p>{{ message }}</p>

  <!-- 속성 바인딩 -->
  <img :src="imageUrl" :alt="imageAlt" />

  <!-- 클래스/스타일 바인딩 -->
  <div :class="{ active: isActive, disabled: isDisabled }"></div>
  <div :style="{ color: textColor, fontSize: size + 'px' }"></div>

  <!-- HTML 직접 렌더링 (XSS 주의) -->
  <div v-html="rawHtml"></div>
</template>
```

### 조건부 렌더링

```vue
<template>
  <p v-if="isLogin">로그인 상태</p>
  <p v-else-if="isPending">대기 중</p>
  <p v-else>로그아웃 상태</p>

  <!-- DOM에서 완전히 제거하지 않고 display:none 처리 -->
  <p v-show="isVisible">토글 표시</p>
</template>
```

> `v-if`는 조건에 따라 DOM을 생성/제거하고, `v-show`는 항상 렌더링된 상태로 `display`만 토글합니다. 자주 토글되는 요소는 `v-show`가 유리합니다.

### 리스트 렌더링

```vue
<template>
  <ul>
    <li v-for="item in items" :key="item.id">
      {{ item.name }}
    </li>
  </ul>

  <!-- index 사용 -->
  <li v-for="(item, index) in items" :key="item.id">
    {{ index }} - {{ item.name }}
  </li>
</template>

<script setup lang="ts">
const items = [
  { id: 1, name: "kim" },
  { id: 2, name: "lee" },
];
</script>
```

> `key`는 반드시 고유 값을 사용해야 하며, `v-for`와 `v-if`를 같은 요소에 함께 사용하지 않습니다 (우선순위 혼동 방지를 위해 `<template>` 래핑 권장).

### 이벤트 처리

```vue
<template>
  <button @click="increase">증가</button>
  <button @click="increaseBy(5)">5씩 증가</button>

  <!-- 이벤트 객체 접근 -->
  <input @keydown.enter="onEnter" />

  <!-- 이벤트 수식어 -->
  <form @submit.prevent="onSubmit">제출</form>
  <div @click.stop="onClick">버블링 방지</div>
</template>

<script setup lang="ts">
import { ref } from "vue";

const count = ref(0);
const increase = () => count.value++;
const increaseBy = (n: number) => (count.value += n);
const onEnter = () => console.log("엔터 입력");
const onSubmit = () => console.log("제출됨");
</script>
```

| 수식어 | 설명 |
|--------|------|
| `.prevent` | `event.preventDefault()` 호출 |
| `.stop` | `event.stopPropagation()` 호출 |
| `.once` | 이벤트 1회만 실행 |
| `.self` | 이벤트가 해당 요소 자체에서 발생했을 때만 실행 |
| `.enter`, `.esc` 등 | 특정 키 입력 시에만 실행 |

### 양방향 바인딩 (v-model)

```vue
<template>
  <input v-model="name" />
  <p>{{ name }}</p>

  <!-- 체크박스 -->
  <input type="checkbox" v-model="checked" />

  <!-- select -->
  <select v-model="selected">
    <option value="a">A</option>
    <option value="b">B</option>
  </select>

  <!-- 수식어 -->
  <input v-model.trim="text" />     <!-- 공백 제거 -->
  <input v-model.number="age" />    <!-- 숫자로 변환 -->
  <input v-model.lazy="text" />     <!-- change 이벤트 시점에 동기화 -->
</template>

<script setup lang="ts">
import { ref } from "vue";

const name = ref("");
const checked = ref(false);
const selected = ref("a");
</script>
```

---

## 3. 반응형 상태 (Reactivity)

### ref vs reactive

| 구분 | `ref` | `reactive` |
|------|-------|-----------|
| 대상 | 원시값·객체 모두 가능 | 객체·배열만 가능 |
| 접근 | `.value` 필요 (템플릿에서는 자동 언랩) | 직접 접근 |
| 재할당 | 가능 (`count.value = 10`) | 객체 자체 교체 불가 |

```ts
import { ref, reactive } from "vue";

// ref - 단일 값 관리
const count = ref(0);
count.value++; // 스크립트에서는 .value 필요

// reactive - 객체 상태 관리
const state = reactive({ count: 0, name: "kim" });
state.count++; // .value 불필요
```

```vue
<template>
  <!-- 템플릿에서는 ref도 .value 없이 자동 언랩 -->
  <p>{{ count }}</p>
  <p>{{ state.count }}</p>
</template>
```

### computed - 계산된 속성

의존하는 값이 바뀔 때만 재계산되며 결과가 캐싱됩니다.

```ts
import { ref, computed } from "vue";

const price = ref(1000);
const quantity = ref(2);

const total = computed(() => price.value * quantity.value);

// 쓰기 가능한 computed
const fullName = computed({
  get: () => `${firstName.value} ${lastName.value}`,
  set: (value) => {
    [firstName.value, lastName.value] = value.split(" ");
  },
});
```

### watch / watchEffect

```ts
import { ref, watch, watchEffect } from "vue";

const count = ref(0);

// 특정 값 감시
watch(count, (newVal, oldVal) => {
  console.log(`${oldVal} → ${newVal}`);
});

// 여러 값 감시
watch([count, name], ([newCount, newName]) => {
  console.log(newCount, newName);
});

// 객체 내부 속성까지 감시 (deep)
watch(state, (newVal) => {
  console.log("state 변경", newVal);
}, { deep: true });

// 즉시 1회 실행 + 이후 감시
watch(count, (val) => console.log(val), { immediate: true });

// watchEffect - 의존성을 자동으로 추적, 즉시 실행
watchEffect(() => {
  console.log(`count: ${count.value}`); // count 사용 시점에 의존성 자동 등록
});
```

| 구분 | `watch` | `watchEffect` |
|------|---------|--------------|
| 의존성 명시 | 명시적으로 지정 | 콜백 내부에서 자동 추적 |
| 초기 실행 | 기본적으로 실행 안 함 (`immediate` 필요) | 즉시 1회 실행 |
| 이전 값 접근 | 가능 | 불가능 |

---

## 4. 컴포넌트

### Props (부모 → 자식)

```vue
<!-- Child.vue -->
<script setup lang="ts">
interface Props {
  title: string;
  count?: number;
}

const props = defineProps<Props>();
// 기본값 설정
const propsWithDefaults = withDefaults(defineProps<Props>(), {
  count: 0,
});
</script>

<template>
  <h2>{{ title }} - {{ count }}</h2>
</template>
```

```vue
<!-- Parent.vue -->
<template>
  <Child title="제목" :count="5" />
</template>

<script setup lang="ts">
import Child from "./Child.vue";
</script>
```

### Emit (자식 → 부모)

```vue
<!-- Child.vue -->
<script setup lang="ts">
const emit = defineEmits<{
  increase: [value: number];
  close: [];
}>();

const handleClick = () => emit("increase", 1);
</script>

<template>
  <button @click="handleClick">증가</button>
  <button @click="emit('close')">닫기</button>
</template>
```

```vue
<!-- Parent.vue -->
<template>
  <Child @increase="onIncrease" @close="onClose" />
</template>

<script setup lang="ts">
const onIncrease = (value: number) => console.log("증가:", value);
const onClose = () => console.log("닫힘");
</script>
```

### v-model을 컴포넌트에서 사용

```vue
<!-- CustomInput.vue -->
<script setup lang="ts">
const props = defineProps<{ modelValue: string }>();
const emit = defineEmits<{ "update:modelValue": [value: string] }>();
</script>

<template>
  <input
    :value="modelValue"
    @input="emit('update:modelValue', ($event.target as HTMLInputElement).value)"
  />
</template>
```

```vue
<!-- Parent.vue -->
<template>
  <CustomInput v-model="text" />
</template>
```

### Slots (콘텐츠 전달)

```vue
<!-- Card.vue -->
<template>
  <div class="card">
    <header><slot name="header">기본 헤더</slot></header>
    <main><slot>기본 내용</slot></main>
    <footer><slot name="footer" :count="5"></slot></footer>
  </div>
</template>
```

```vue
<!-- 사용 -->
<template>
  <Card>
    <template #header>커스텀 헤더</template>
    본문 내용
    <template #footer="{ count }">총 {{ count }}개</template>
  </Card>
</template>
```

---

## 5. 생명주기 훅 (Lifecycle Hooks)

```vue
<script setup lang="ts">
import {
  onMounted,
  onUpdated,
  onUnmounted,
  onBeforeMount,
  onBeforeUnmount,
} from "vue";

onBeforeMount(() => console.log("마운트 직전"));
onMounted(() => console.log("마운트 완료 - DOM 접근 가능"));
onUpdated(() => console.log("리렌더링 완료"));
onBeforeUnmount(() => console.log("언마운트 직전 - cleanup"));
onUnmounted(() => console.log("언마운트 완료"));
</script>
```

| 훅 | 시점 | 주요 용도 |
|----|------|----------|
| `onMounted` | DOM 마운트 완료 후 | API 호출, DOM 직접 접근 |
| `onUpdated` | 반응형 데이터 변경 후 리렌더링 완료 | DOM 업데이트 후처리 |
| `onBeforeUnmount` | 컴포넌트 제거 직전 | 타이머·이벤트 리스너 해제 |
| `onUnmounted` | 컴포넌트 제거 완료 | 리소스 정리 확인 |

---

## 6. Composables (재사용 로직)

React의 custom hook과 유사하게, `use` 접두사 함수로 로직을 재사용합니다.

```ts
// composables/useCounter.ts
import { ref } from "vue";

export function useCounter(initial = 0) {
  const count = ref(initial);
  const increase = () => count.value++;
  const decrease = () => count.value--;
  return { count, increase, decrease };
}
```

```vue
<script setup lang="ts">
import { useCounter } from "@/composables/useCounter";

const { count, increase, decrease } = useCounter(10);
</script>

<template>
  <p>{{ count }}</p>
  <button @click="increase">+</button>
  <button @click="decrease">-</button>
</template>
```

```ts
// composables/useFetch.ts - 비동기 데이터 패칭 예시
import { ref } from "vue";

export function useFetch<T>(url: string) {
  const data = ref<T | null>(null);
  const error = ref<string | null>(null);
  const loading = ref(true);

  fetch(url)
    .then((res) => res.json())
    .then((json) => (data.value = json))
    .catch((err) => (error.value = err.message))
    .finally(() => (loading.value = false));

  return { data, error, loading };
}
```

---

## 7. 상태 관리 (Pinia)

Vuex의 후속 표준 상태 관리 라이브러리입니다.

```bash
npm install pinia
```

```ts
// main.ts
import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";

createApp(App).use(createPinia()).mount("#app");
```

```ts
// stores/counter.ts
import { defineStore } from "pinia";

export const useCounterStore = defineStore("counter", {
  state: () => ({
    count: 0,
  }),
  getters: {
    doubleCount: (state) => state.count * 2,
  },
  actions: {
    increase() {
      this.count++;
    },
    async fetchInitial() {
      const res = await fetch("/api/count");
      this.count = await res.json();
    },
  },
});
```

```vue
<!-- 컴포넌트에서 사용 -->
<script setup lang="ts">
import { useCounterStore } from "@/stores/counter";

const counterStore = useCounterStore();
</script>

<template>
  <p>{{ counterStore.count }} / 2배: {{ counterStore.doubleCount }}</p>
  <button @click="counterStore.increase">+</button>
</template>
```

```ts
// Composition API 스타일 store
export const useCounterStore = defineStore("counter", () => {
  const count = ref(0);
  const doubleCount = computed(() => count.value * 2);
  const increase = () => count.value++;
  return { count, doubleCount, increase };
});
```

---

## 8. 라우팅 (Vue Router)

```bash
npm install vue-router
```

```ts
// router/index.ts
import { createRouter, createWebHistory } from "vue-router";
import Home from "@/views/Home.vue";
import About from "@/views/About.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: Home },
    { path: "/about", component: About },
    { path: "/users/:id", component: () => import("@/views/UserDetail.vue") }, // 지연 로딩
  ],
});

export default router;
```

```ts
// main.ts
import router from "./router";
createApp(App).use(router).mount("#app");
```

```vue
<template>
  <RouterLink to="/about">About</RouterLink>
  <RouterView />
</template>

<script setup lang="ts">
import { useRouter, useRoute } from "vue-router";

const router = useRouter();
const route = useRoute();

const goToAbout = () => router.push("/about");
const userId = route.params.id;
</script>
```

### 네비게이션 가드

```ts
router.beforeEach((to, from, next) => {
  if (to.meta.requiresAuth && !isLoggedIn()) {
    next("/login");
  } else {
    next();
  }
});
```

---

## 9. 환경 변수

```bash
# .env.local / .env.development / .env.production
# Vite는 VITE_ 접두사 필수
VITE_API_URL=https://api.example.com
```

```ts
const apiUrl = import.meta.env.VITE_API_URL as string;
```

### 빌드

```bash
npm run build   # dist/ 폴더에 빌드 결과 생성
npx serve dist  # 빌드 결과 로컬 실행 테스트
```

---

## 10. Vue 3 vs Vue 2 / Composition API vs Options API

| 구분 | Options API (Vue 2) | Composition API (Vue 3) |
|------|---------------------|--------------------------|
| 로직 구성 | `data`, `methods`, `computed` 분리 | `setup()` 또는 `<script setup>`에 통합 |
| 재사용 | Mixin (이름 충돌 위험) | Composables (함수 단위, 충돌 없음) |
| 타입 추론 | 약함 | 강함 (TypeScript 친화적) |
| 코드 응집도 | 기능별로 분산 | 관련 로직이 한곳에 모임 |

```vue
<!-- Options API -->
<script>
export default {
  data() {
    return { count: 0 };
  },
  methods: {
    increase() {
      this.count++;
    },
  },
};
</script>

<!-- Composition API (script setup) -->
<script setup>
import { ref } from "vue";
const count = ref(0);
const increase = () => count.value++;
</script>
```

---

## 11. 주요 이벤트/패턴 모음

```vue
<script setup lang="ts">
import { ref, watch } from "vue";

// debounce (검색 입력 최적화)
function useDebounce<T>(value: import("vue").Ref<T>, delay: number) {
  const debounced = ref(value.value) as import("vue").Ref<T>;
  let timer: ReturnType<typeof setTimeout>;

  watch(value, (newVal) => {
    clearTimeout(timer);
    timer = setTimeout(() => (debounced.value = newVal), delay);
  });

  return debounced;
}

const text = ref("");
const debouncedText = useDebounce(text, 500);

// null/undefined 방어
const safeName = (name?.value ?? "").trim();
</script>

<template>
  <!-- 파일 업로드 -->
  <input type="file" @change="(e) => console.log((e.target as HTMLInputElement).files?.[0])" />

  <!-- 동적 컴포넌트 -->
  <component :is="currentView" />

  <!-- Teleport (모달을 body 최상단으로 렌더링) -->
  <Teleport to="body">
    <div class="modal">모달 내용</div>
  </Teleport>

  <!-- Suspense (비동기 컴포넌트 로딩 처리) -->
  <Suspense>
    <template #default><AsyncComponent /></template>
    <template #fallback>로딩 중...</template>
  </Suspense>
</template>
```

---

## 12. 외부 JS 파일 적용

```html
<!-- index.html에 직접 추가 -->
<script src="/js/solution.js" defer></script>
```

```vue
<!-- onMounted에서 동적 로드 -->
<script setup lang="ts">
import { onMounted } from "vue";

onMounted(() => {
  if (document.getElementById("map-script")) return;

  const script = document.createElement("script");
  script.id = "map-script";
  script.src = "https://cdn.example.com/map.js";
  script.async = true;
  script.onload = () => {
    (window as any).initMap();
  };
  document.body.appendChild(script);
});
</script>
```

```ts
// types/global.d.ts - 외부 스크립트의 전역 변수 타입 선언
declare global {
  interface Window {
    kakao: any;
    initMap: () => void;
  }
}
export {};
```
