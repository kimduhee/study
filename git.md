# 저장소 생성/가져오기
### 새 Git 저장소 생성
+ 현재 폴더를 Git 저장소로 초기화
<pre><code>git init</code></pre>

### 원격 저장소 가져오기
<pre><code>git clont [repository]
  
예)git clone https://github.com/kimduhee/study.git
</code></pre>


# 변경사항 확인
### 현재 상태 확인
+ 수정된 파일
+ staging 된 파일
+ commit 전 파일 확인
<pre><code>git status</code></pre>

### 변경 내용 보기
+ staging 된 것 확인
<pre><code>git diff
git diff --staged
</code></pre>


# 파일 추가(Staging)
### 특정 파일 추가
<pre><code>git file.txt</code></pre>

### 전체 파일 추가
<pre><code>git add .</code></pre>

### 변경된 파일만 추가
<pre><code>git add -u</code></pre>


# Commit
### commit
<pre><code>git commit -m "init commit"</code></pre>

### add + commit
<pre><code>git commit -am "init commit"</code></pre>


# 원격 저장소 작업
### 원격 저장소 확인
<pre><code>git remote -v</code></pre>

### 원격 저장소 추가
<pre><code>git remote add origin [url]</code></pre>

### 원격 저장소 삭제
<pre><code>git remote remove origin</code></pre>

### push
<pre><code>git push origin main</code></pre>
+ 최초 push
<pre><code>git push -u origin main</code></pre>

### pull
<pre><code>git pull</code></pre>


# 브랜치
### 브랜치 목록
<pre><code>git branch</code></pre>

### 브랜치 생성
<pre><code>git branch feature/test</code></pre>

### 브랜치 이동
<pre><code>git checkout feature/function</code></pre>
+최근 많이 쓰는 방식
<pre><code>git switch feature/function</code></pre>

### 브랜치 생성 + 이동
<pre><code>git checkout -b feature/test</code></pre>


# Merge
### 현재 브랜치에 merge
<pre><code>git merge [branch name]

예)git merge dev
</code></pre>

### 원격 정보만 가져오기
<pre><code>git fetch</code></pre>


# 로그 확인
### commit 로그
<pre><code>git log</code></pre>

### 한 줄 보기
<pre><code>git log --oneline</code></pre>

### 그래프 보기
<pre><code>git log --oneline --graph</code></pre>


# 되돌리기
### staging 취소
<pre><code>git restore --staged file.txt</code></pre>

### 파일 수정 취소
<pre><code>git restore file.txt</code></pre>

### 마지막 commit 취소(코드는 유지)
+ commit 삭제
+ 히스토리 변경
<pre><code>git reset --soft HEAD~1</code></pre>

### commit + 코드 삭제
+ 위험 요소 있음
<pre><code>git reset --hard HEAD~1</code></pre>

### revert
+ commit 취소 commit 생성
+ 협업에 안전
<pre><code>git revert [commit-id]</code></pre>

### 최근 commit 상세 보기
<pre><code>git show</code></pre>


# stash(작업 임시 저장)
### stash
<pre><code>git stash</code></pre>

### stash 목록
<pre><code>git stash list</code></pre>

### stash 복구
<pre><code>git stash pop</code></pre>
