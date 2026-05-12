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
<pre><code><button @click="increase">
  증가
</button>

<script setup>
import { ref } from 'vue'

const count = ref(0)

const increase = () => {
  count.value++
}
</script>
</code></pre>
