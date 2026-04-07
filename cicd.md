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
+ Executor: shell
+ description: local-windows-runner
+ optional: test runner for React CI/CD
+ tags: X

Runner 삭제
<pre><code>gitlab-runner unregister --all-runners</code></pre>

Runners 확인
+ Status [online] 확인

| <img width="1303" height="650" alt="Image" src="https://github.com/user-attachments/assets/815ff233-22b0-4efe-991b-9d6e5ba51874" /> |
|---------------|

### pipelines 실행 결과
| <img width="1331" height="489" alt="Image" src="https://github.com/user-attachments/assets/eb20c1fe-6936-43e2-a517-2ed0e4f481b1" /> |
|---------------|

| <img width="1303" height="650" alt="Image" src="https://github.com/user-attachments/assets/815ff233-22b0-4efe-991b-9d6e5ba51874" /> |
|---------------|

### Vartifacts 설정
프로젝트 > Settings > CI/CD > Variable
+ CI_USERNAME: git 계정 아이디
+ CI_PERSONAL_TOKEN: git 계정 비번
+ GIT_STRATEGY: none

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

