# gitlab cicd
### docker desktop 설치
URL: https://www.docker.com/
+ Use WSL 2 instead of Hyper-V(recommentded) 체크 후 설치 

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
