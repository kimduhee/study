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

### root 계정 초기 비밀번호 확인
GitLab 최초 설치 시 root 계정 비밀번호는 컨테이너 내부 파일에 자동 생성됨 (24시간 후 파일 삭제됨)
<pre><code>docker exec -it gitlab grep 'Password:' /etc/gitlab/initial_root_password</code></pre>

---

### .gitlab-ci.yml 기본 구조
프로젝트 루트에 `.gitlab-ci.yml` 파일을 생성하면 push 시 자동으로 파이프라인이 실행됨
<pre><code>stages:
  - build
  - test
  - deploy

build-job:
  stage: build
  image: node:18
  script:
    - npm install
    - npm run build
  artifacts:
    paths:
      - dist/

test-job:
  stage: test
  image: node:18
  script:
    - npm install
    - npm run test

deploy-job:
  stage: deploy
  image: node:18
  script:
    - echo "배포 진행"
  only:
    - main
</code></pre>
> stages: 파이프라인 단계 순서 정의 (같은 stage의 job은 병렬 실행)<br>
> image: Docker Executor가 사용할 이미지 (job별로 다르게 지정 가능)<br>
> script: 실제 실행할 명령어<br>
> artifacts: 다음 stage로 전달할 결과물(빌드 산출물)<br>
> only/rules: 특정 브랜치·조건에서만 job 실행

### before_script / after_script
모든 job 실행 전후 공통으로 처리할 작업 정의
<pre><code>default:
  before_script:
    - echo "작업 시작 전 공통 처리"
  after_script:
    - echo "작업 종료 후 공통 처리"
</code></pre>

---

### CI/CD 변수 실제 사용 예시
Settings > CI/CD > Variable에 등록한 값은 `$변수명`으로 파이프라인에서 사용 가능
<pre><code>checkout-job:
  stage: build
  image: alpine/git
  script:
    - git clone https://$CI_USERNAME:$CI_PERSONAL_TOKEN@gitlab.local/group/project.git .
    - echo "clone 완료"
</code></pre>
> CI_USERNAME, CI_PERSONAL_TOKEN을 코드에 평문으로 노출하지 않고 변수로 분리 관리<br>
> Variable 등록 시 "Protect variable", "Mask variable" 옵션으로 로그 노출 방지 가능

### GIT_STRATEGY 옵션
GitLab CI는 기본적으로 job 시작 전 자동으로 git clone(또는 fetch)을 수행함

| 값 | 설명 |
|----|------|
| `clone` | 매 job마다 전체 새로 clone (기본값, 가장 안전하지만 느림) |
| `fetch` | 기존 워크스페이스 유지하며 변경분만 fetch (빠름) |
| `none` | 자동 git 동작 없음 → script에서 직접 clone 처리 필요 |
| `empty` | 워크스페이스를 비움 |

<pre><code>variables:
  GIT_STRATEGY: fetch
</code></pre>
> 본인 실습에서는 GIT_STRATEGY: none으로 설정 후 위 checkout-job처럼 토큰 인증 git clone을 직접 수행

---

### cache vs artifacts
같은 파이프라인 내 산출물 공유 방식이지만 목적이 다름

| 구분 | cache | artifacts |
|------|-------|-----------|
| 목적 | 빌드 속도 향상 (의존성 재사용) | stage 간 결과물 전달 |
| 보존 위치 | Runner 로컬 | GitLab 서버에 업로드 |
| 대표 대상 | `node_modules/`, `.m2/` | `dist/`, 빌드된 jar/war |
| 파이프라인 종료 후 | Runner에 남아 재사용 | 만료 기간 후 자동 삭제 |

<pre><code>build-job:
  stage: build
  script:
    - npm install
    - npm run build
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/
  artifacts:
    paths:
      - dist/
    expire_in: 1 week
</code></pre>

---

### rules로 실행 조건 제어
`only`/`except`보다 최신 권장 방식, 조건 조합이 자유로움
<pre><code>deploy-job:
  stage: deploy
  script:
    - echo "운영 배포"
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: on_success
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
      when: never
    - when: never
</code></pre>

### stage 간 의존성 (needs)
기본적으로 다음 stage는 이전 stage 전체 완료를 기다리지만, `needs`로 특정 job만 의존하도록 지정해 대기 시간을 줄일 수 있음
<pre><code>test-job:
  stage: test
  needs: ["build-job"]
  script:
    - npm run test
</code></pre>

---

### Docker in Docker (DinD) - 이미지 빌드가 필요한 경우
Docker Executor 컨테이너 안에서 또 Docker 이미지를 빌드(`docker build`)하려면 별도 설정 필요
<pre><code>build-image-job:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  script:
    - docker build -t my-app:latest .
    - docker push my-app:latest
</code></pre>
> gitlab-runner register 시 Executor를 docker로 선택했다면, Job에서 docker 명령어를 쓰려면 `docker:dind` 서비스를 함께 등록해야 함<br>
> Runner의 config.toml에 `privileged = true` 설정 필요 (보안 주의)

---

### 트러블슈팅 (실습 중 자주 겪는 문제)

**1. Runner가 GitLab(host.docker.internal)에 연결되지 않음**
> Windows/Mac Docker Desktop은 `host.docker.internal`을 자동 지원하지만, Linux 환경에서는 미지원<br>
> Linux에서는 `--add-host=host.docker.internal:host-gateway` 옵션을 docker run에 추가하거나 실제 호스트 IP 사용

**2. Runner는 등록됐지만 Job이 계속 pending 상태**
> Runner의 tags와 .gitlab-ci.yml의 tags 설정이 불일치하는 경우 발생<br>
> 등록 시 tags를 비워뒀다면(`Runner runs untagged jobs` 체크) job에도 tags를 지정하지 않아야 매칭됨

**3. GIT_STRATEGY: none 설정 후 파일이 없다는 오류**
> 자동 clone이 꺼졌으므로 script 첫 줄에서 직접 git clone을 호출해야 함 (위 checkout-job 참고)

**4. self-signed 인증서로 인한 git clone SSL 오류**
> 로컬 GitLab은 기본적으로 HTTP(8080)로 동작하므로 SSL 오류 시 URL이 https로 잘못 지정됐는지 확인

**5. Docker Executor에서 docker 명령어 실행 시 권한 오류**
> DinD 서비스 미설정 또는 Runner `privileged` 옵션 누락이 원인인 경우가 많음

---

### Runner 동시 실행 설정
`config.toml`(C:/gitlab-runner/config.toml)에서 동시 처리 가능한 job 수 조정
<pre><code>concurrent = 4

[[runners]]
  name = "local-windows-runner"
  executor = "docker"
  [runners.docker]
    image = "node:18"
    privileged = true
</code></pre>
> concurrent: 전체 Runner가 동시에 처리할 수 있는 최대 job 수<br>
> 등록된 여러 Runner 또는 하나의 Runner로 여러 job을 병렬 처리할 때 조정

