# gitlab cicd
### docker desktop 설치
+ URL: https://www.docker.com/
+ Use WSL 2 instead of Hyper-V(recommentded) 체크 후 설치 


### gitlab runner 설치
+ URL: https://docs.gitlab.com/runner/install/windows/#download-and-install-gitlab-runner
+ gitlab-runner-windows-amd64.exe 파일 다운로드 후 C:/gitlab-runner 폴더 생성후 다운로드 받은 파일을 해당 폴더에 추가
+ 파일 이름을 편의상 gitlab-runner.exe로 변경
+ 시스템 환경변수 path에 C:/gitlab-runner 추가

설치 확인
<pre><code>gitlab-runner --version</code></pre>

Runner 등록
<pre><code>gitlab-runner register</code></pre>
+ GitLab URL: http://host.docker.internal:8080/
+ Registration token → 프로젝트 → Settings → CI/CD → Runners → Registration token
+ Executor: docker
+ description: local-windows-runner
+ optional: test runner for React CI/CD
+ tags: X
+ default Docker image: default Docker image

Runner 확인
<pre><code>gitlab-runner list</code></pre>

Runner 삭제
<pre><code>gitlab-runner unregister --all-runners</code></pre>

Runners 확인
+ Status [online] 확인

| <img width="1303" height="650" alt="Image" src="https://github.com/user-attachments/assets/815ff233-22b0-4efe-991b-9d6e5ba51874" /> |
|---------------|

### pipelines 실행 결과
| <img width="1331" height="489" alt="Image" src="https://github.com/user-attachments/assets/eb20c1fe-6936-43e2-a517-2ed0e4f481b1" /> |
|---------------|

| <img width="1254" height="418" alt="Image" src="https://github.com/user-attachments/assets/3d58ee6f-e340-4eb3-926b-dcad8b95cbc4" /> |
|---------------|

### Variables 설정
프로젝트 > Settings > CI/CD > Variable
+ CI_USERNAME: git 계정 아이디
+ CI_PERSONAL_TOKEN: git 계정 비번
+ GIT_STRATEGY: none => GitLab CI는 job 시작 전에 자동으로 git clone를 하는데 수동으로 처리 하기 위해 기능 끔

### gitlab 설치
<pre><code>docker run -d \
  --hostname gitlab.local \
  --publish 8080:80 --publish 2222:22 \
  --name gitlab \
  --restart always \
  -v gitlab_config:/etc/gitlab \
  -v gitlab_logs:/var/log/gitlab \
  -v gitlab_data:/var/opt/gitlab \
  gitlab/gitlab-ee
</code></pre>
+ 접속: http://localhost:8080

### Docker Executor와 Shell Executor
Docker Executor와 Shell Executor는 대표적인 CI/CD 도구인 GitLab CI에서 작업(Job)을 실행하는 방식이며
어디서, 어떻게 실행되느냐가 핵심 차이이다.

#### Docker Executor<br>
+ 특징
>-Docker Executor는 각 Job을 컨테이너 안에서 격리된 환경으로 실행<br>
>-Docker 컨테이너 기반 실행<br>
>-실행할 때마다 새로운 컨테이너 생성 → 깨끗한 환경<br>
>-.gitlab-ci.yml에서 이미지 지정 (python:3.11, node:18 등)<br>
>-의존성 충돌 없음 (환경이 독립적)

+ 장점
>-환경 재현성 좋음 (어디서 실행해도 동일)<br>
>-의존성 관리 쉬움<br>
>-보안성 높음 (격리됨)

+ 단점
>-Docker 설치 및 설정 필요<br>
>-컨테이너 생성 → 약간의 오버헤드<br>
>-Docker-in-Docker 같은 추가 설정이 필요할 수 있음

#### Shell Executor<br>
+ 특징
>-서버(러너)에 설치된 환경 그대로 사용<br>
>-bash, sh, powershell 등에서 실행<br>
>-별도의 컨테이너 없음

+ 장점
>-빠름 (컨테이너 생성 없음)<br>
>-설정 간단<br>
>-시스템 자원 직접 활용 가능

+ 단점
>-환경 오염 가능성(패키지 충돌)<br>
>-재현성 낮음 (서버 상태에 의존)<br>
>-보안 취약 (격리 없음)

### 기타 명령어
> 기존 컨테이너 확인<br>
> <pre><code>docker ps -a</code></pre>
>
> 컨테이너(gitlab) 중지<br>
> <pre><code>docker stop gitlab</code></pre>
>
> 컨테이너(gitlab) 삭제<br>
> <pre><code>docker rm gitlab</code></pre>
>
> 컨테이너(gitlab) 시작<br>
> <pre><code>docker start gitlab</code></pre>
>
> 컨테이너(gitlab) 로그 확인<br>
> <pre><code>docker logs -f gitlab</code></pre>

