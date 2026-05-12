# 환경 구성
### Node.js 설치
<pre><code>node -v
npm -v
</code></pre>

### Vite 프로젝트 생성
<pre><code>npm create vite@latest
</code></pre>
> -프로젝트 이름: 본인의 프로젝트명(vue-project)<br>
> -프레임워크 선택: Vue<br>
> -Variant 선택: TypeScript


# 문법
### click 이벤트
<pre><code>&lt;button @click="increase">증가&lt;/button>

&lt;script setup>
import { ref } from 'vue'

const count = ref(0)

const increase = () => {
  count.value++
}
&lt;/script>
</code></pre>

### 조건
<pre><code>&lt;p v-if="isLogin">로그인 상태&lt;/p>
&lt;p v-else>로그아웃 상태&lt;/p>
</code></pre>

### 반복문
<pre><code>&lt;template>
  &lt;div v-for="item in items" :key="item.id">
    {{ item.name }}
  &lt;/div>
&lt;/template>

&lt;script setup>
const items = [
  { id: 1, name: 'kim' },
  { id: 2, name: 'lee' }
]
&lt;/script>
</code></pre>

### 양방향 바인딩
<pre><code>&lt;input v-model="name" />

&lt;p>{{ name }}&lt;/p>

&lt;script setup>
import { ref } from 'vue'

const name = ref('')
&lt;/script>
</code></pre>
