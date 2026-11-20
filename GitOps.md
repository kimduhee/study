# 로컬 ArgoCD GitOps 실습 (Windows + Docker Desktop + kind + Gitea)

## 구성 개요

| 구성요소 | 위치 | 역할 |
|---|---|---|
| Docker Desktop (WSL2) | 로컬 PC | 컨테이너 런타임 |
| kind 클러스터 `dev` | Docker 컨테이너 | 로컬 k8s |
| ArgoCD | kind 클러스터 내부 (`argocd` ns) | Git → 클러스터 동기화 |
| Gitea | k8s 밖, Docker 컨테이너 | 경량 Git 서버 (SQLite) |
| demo-app | kind 클러스터 (`demo` ns) | Spring Boot 앱 |

- Git 서버는 k8s 안에 넣지 않고 Docker 컨테이너로 분리한다.
- 우선 개발(dev) 클러스터만 구성하고, 운영(prod)은 나중에 별도 클러스터 + 별도 ArgoCD로 추가한다.
- 모든 명령은 PowerShell 기준.


```
[소스 수정] → 빌드 → 이미지 태그 갱신 → 매니페스트 repo push → ArgoCD가 감지 → 클러스터 반영
                                       (Gitea)                (ArgoCD 담당 구간)
```

---

## 0. 자원 제한 (16GB PC)

`C:\Users\<계정>\.wslconfig`

```
[wsl2]
memory=6GB
processors=4
```
> WSL2에 할당할 최대 메모리와 CPU 수를 지정한다. Docker Desktop이 WSL2 백엔드를 사용하므로 여기서 자원을 제한하면 Docker 전체에 적용된다.

```powershell
wsl --shutdown    # WSL2 인스턴스를 완전히 종료한 뒤 Docker Desktop을 재시작해야 설정이 반영된다
```

Docker Desktop은 WSL2 백엔드여야 한다.

## 1. CLI 설치

```powershell
# kubectl: 쿠버네티스 클러스터를 제어하는 공식 CLI 도구
winget install Kubernetes.kubectl

# kind (Kubernetes IN Docker): Docker 컨테이너를 노드로 사용하는 로컬 k8s 클러스터 도구
winget install Kubernetes.kind

# argocd: ArgoCD 서버를 CLI로 제어하는 도구 (저장소 등록, 앱 동기화 등)
# winget 패키지가 없으면 GitHub releases에서 argocd-windows-amd64.exe를 다운로드해 PATH에 추가
winget install argoproj.argocd
```

## 2. kind 클러스터 생성

```powershell
# 'dev'라는 이름의 로컬 k8s 클러스터를 생성한다
# 내부적으로 Docker 컨테이너(dev-control-plane) 하나가 k8s 노드로 동작한다
kind create cluster --name dev

# kubectl이 'dev' 클러스터를 바라보도록 현재 컨텍스트를 변경한다
# 여러 클러스터를 사용할 때 어느 클러스터에 명령을 보낼지 지정하는 개념이 컨텍스트
kubectl config use-context kind-dev

# 클러스터의 노드 목록을 조회해 정상 생성 여부를 확인한다 (STATUS: Ready)
kubectl get nodes
```

kind는 Docker 네트워크 `kind`를 자동 생성한다. 3단계의 Gitea를 이 네트워크에 붙인다.

## 3. Gitea 실행 (k8s 밖)

`C:\gitops\gitea\docker-compose.yml`

```yaml
services:
  gitea:
    image: gitea/gitea:latest
    container_name: gitea
    environment:
      # DB를 별도로 띄우지 않고 SQLite 파일 하나로 운영 (로컬 실습용)
      - GITEA__database__DB_TYPE=sqlite3
      # 외부 접속 시 사용할 루트 URL (clone/push URL 생성에 사용됨)
      - GITEA__server__ROOT_URL=http://localhost:3000/
    ports:
      # 호스트 3000 포트를 컨테이너 3000 포트로 연결 (웹 UI 및 HTTP Git 접근)
      - "3000:3000"
    volumes:
      # Gitea 데이터(저장소, 설정, DB 파일)를 호스트 경로에 영속 저장
      - ./data:/data
    restart: unless-stopped
```

```powershell
# docker-compose.yml이 있는 폴더로 이동
cd C:\gitops\gitea

# 백그라운드(-d)로 Gitea 컨테이너를 실행한다
docker compose up -d

# Gitea 컨테이너를 kind 네트워크에 연결한다
# ArgoCD 파드(클러스터 내부)가 컨테이너명 'gitea'로 Git 서버에 접근하려면
# 같은 Docker 네트워크에 있어야 한다
docker network connect kind gitea
```

초기 설정:
1. `http://localhost:3000` 접속
2. 기본값 유지, 맨 아래 관리자 계정(아이디/비번/이메일) 입력 후 설치
3. 로그인 후 `+` → 새 저장소 → 이름 `demo`

확인:

```powershell
# 실행 중인 모든 컨테이너 목록을 확인한다 (STATUS: Up이면 정상)
docker ps

# kind 네트워크에 연결된 컨테이너 목록을 확인한다
# "Containers" 항목에 gitea와 dev-control-plane이 모두 있어야
# 클러스터 내부에서 Gitea에 접근할 수 있다
docker network inspect kind
```

> ArgoCD(클러스터 내부)에서 `localhost`는 파드 자신을 가리킨다. 그래서 이후 저장소 주소는 컨테이너명 `http://gitea:3000`을 쓴다.

## 4. ArgoCD 설치

```powershell
# argocd 네임스페이스를 생성한다
# 네임스페이스는 k8s 내에서 리소스를 논리적으로 격리하는 단위
kubectl create namespace argocd

# ArgoCD 공식 매니페스트를 클러스터에 적용한다
# --server-side: 서버 측에서 적용(충돌 방지, CRD 등 대형 매니페스트에 권장)
# -f: 적용할 매니페스트 파일(URL)을 지정
kubectl apply -n argocd --server-side -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# argocd 네임스페이스의 파드 상태를 실시간 감시(-w, watch)한다
# 7개 파드가 모두 1/1 Running이면 설치 완료 → Ctrl+C로 감시 종료
kubectl -n argocd get pods -w
```

UI 접속 (이 창은 계속 열어둔다):

```powershell
# argocd-server 서비스의 443 포트를 로컬 8080 포트로 포워딩한다
# 이 창을 닫으면 접속이 끊기므로 별도 터미널에서 유지해야 한다
kubectl -n argocd port-forward svc/argocd-server 8080:443
```

새 PowerShell 창에서 초기 비밀번호 확인:

```powershell
# ArgoCD CLI를 이용해 초기 admin 비밀번호를 조회한다
argocd admin initial-password -n argocd

# 또는 kubectl로 Secret에서 직접 base64 디코딩해서 비밀번호를 얻는다
# -o jsonpath: JSON 경로 표현식으로 특정 필드만 추출
# [Convert]::FromBase64String: base64 → 바이트 배열 변환
# [Text.Encoding]::UTF8.GetString: 바이트 배열 → 문자열 변환
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | % { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }
```

`https://localhost:8080` (인증서 경고 무시), 계정 `admin`.

## 5. 앱 이미지 + 매니페스트 준비

### A. jar 빌드 + 이미지 빌드 + kind 로드

```powershell
# Maven Wrapper로 프로젝트를 빌드한다
# clean: 이전 빌드 결과물 삭제
# package: 컴파일 + 테스트 + jar 패키징
# -DskipTests: 테스트 실행 생략 (빌드 속도 향상)
# Gradle 프로젝트라면: .\gradlew bootJar
.\mvnw clean package -DskipTests
```

`Dockerfile` (프로젝트 루트)

```dockerfile
# 경량 JRE 21 기반 이미지 (JDK가 아닌 JRE라 이미지 크기 작음)
FROM eclipse-temurin:21-jre

# 빌드된 jar 파일을 컨테이너의 /app.jar로 복사
# Gradle: build/libs/*.jar
COPY target/*.jar app.jar

# 컨테이너가 사용하는 포트를 문서화 (실제 포트 개방은 docker run -p로 함)
EXPOSE 8080

# 컨테이너 시작 시 실행할 명령어 (java -jar /app.jar)
ENTRYPOINT ["java","-jar","/app.jar"]
```

```powershell
# demo-app:0.1 태그로 Docker 이미지를 빌드한다
# -t: 이미지 이름:태그 지정
# .: 현재 디렉토리의 Dockerfile을 사용
docker build -t demo-app:0.1 .

# 빌드한 이미지를 kind 클러스터 내부로 로드한다
# kind 클러스터는 외부 레지스트리에 접근하지 않으므로 직접 주입이 필요
# --name dev: 이미지를 로드할 클러스터 이름
kind load docker-image demo-app:0.1 --name dev

# 클러스터 노드(dev-control-plane) 컨테이너에 접속해 이미지 로드 여부를 확인한다
# crictl: 컨테이너 런타임 제어 CLI (k8s 내부용)
# findstr: Windows의 grep 역할 (특정 문자열이 포함된 줄만 출력)
docker exec -it dev-control-plane crictl images | findstr demo-app
```

- 레지스트리 없이 `kind load`로 클러스터에 직접 이미지를 넣는다.
- 태그를 `latest`로 쓰면 안 된다 (항상 pull을 시도해서 실패).

### B. 매니페스트 작성 (앱 소스와 별도 폴더)

```powershell
# 매니페스트 전용 폴더를 생성한다 (앱 소스와 분리해야 GitOps 흐름이 명확해짐)
mkdir C:\gitops\demo\k8s
cd C:\gitops\demo
```

`k8s\deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment              # 파드를 관리하는 리소스 종류
metadata:
  name: demo-app
spec:
  replicas: 1                 # 실행할 파드 복제본 수
  selector:
    matchLabels:
      app: demo-app           # 이 Deployment가 관리할 파드를 label로 선택
  template:
    metadata:
      labels:
        app: demo-app         # 파드에 붙는 레이블 (selector와 일치해야 함)
    spec:
      containers:
        - name: demo-app
          image: demo-app:0.1
          # IfNotPresent: 이미지가 노드에 없을 때만 pull (kind load로 넣었으므로 로컬 사용)
          # Always로 설정하면 외부 레지스트리에서 pull 시도 → 실패
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080   # 컨테이너가 실제로 열 포트
```

`k8s\service.yaml`

```yaml
apiVersion: v1
kind: Service                 # 파드에 대한 네트워크 접근 방법을 정의하는 리소스
metadata:
  name: demo-app
spec:
  selector:
    app: demo-app             # 트래픽을 보낼 파드를 label로 선택
  ports:
    - port: 80                # Service가 외부에 노출하는 포트
      targetPort: 8080        # 트래픽을 전달할 파드의 실제 포트
  # type 미지정 시 기본값 ClusterIP: 클러스터 내부에서만 접근 가능
```

앱 포트가 8080이 아니면 `containerPort`, `targetPort`를 맞춘다.

### C. Gitea에 push

```powershell
cd C:\gitops\demo

# 현재 폴더를 Git 저장소로 초기화하고 기본 브랜치를 main으로 설정한다
git init -b main

# 모든 파일을 스테이징 영역에 추가한다
git add .

# 첫 번째 커밋을 생성한다
git commit -m "init manifests"

# 로컬 저장소에 Gitea 원격 저장소 주소를 'origin'으로 등록한다
git remote add origin http://localhost:3000/<유저>/demo.git

# 로컬 main 브랜치를 origin/main으로 처음 push하고 추적 관계를 설정한다
# -u: upstream 설정 (이후 git push만 해도 origin main으로 push됨)
git push -u origin main
```

## 6. ArgoCD에 저장소 + Application 등록

### 로그인 (CLI 사용 전 필수)

port-forward 창이 열려 있는 상태에서:

```powershell
# ArgoCD CLI를 서버에 인증한다
# localhost:8080: port-forward로 연결된 ArgoCD 서버 주소
# --username admin: 관리자 계정으로 로그인
# --insecure: 자체 서명 인증서 검증 생략 (로컬 실습용)
# 로그인 성공 시 ~/.config/argocd/config 에 세션 토큰이 저장됨
argocd login localhost:8080 --username admin --insecure
```

### 저장소 등록

```powershell
# ArgoCD가 폴링할 Git 저장소를 등록한다
# gitea:3000: kind 네트워크 내에서 컨테이너명으로 접근 (localhost가 아님에 주의)
# --username, --password: Gitea 계정 자격증명 (ArgoCD가 clone 시 사용)
argocd repo add http://gitea:3000/<유저>/demo.git --username <유저> --password <비번>

# 등록된 저장소 목록과 연결 상태를 확인한다 (STATUS: Successful이면 정상)
argocd repo list
```

CLI가 막히면 UI: Settings → Repositories → Connect Repo (VIA HTTPS / git / URL 입력 / Gitea 계정).

### Application 생성

`C:\gitops\demo\application.yaml` — `k8s` 폴더 **밖**에 둔다 (안에 두면 ArgoCD가 자기 자신을 배포 대상으로 읽는다).

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application               # ArgoCD 전용 커스텀 리소스(CRD)
metadata:
  name: demo
  namespace: argocd             # ArgoCD가 설치된 네임스페이스에 생성해야 함
spec:
  project: default              # ArgoCD 프로젝트 (권한/정책 묶음); default는 제한 없음
  source:
    repoURL: http://gitea:3000/<유저>/demo.git   # 감시할 Git 저장소 URL
    targetRevision: main                          # 감시할 브랜치/태그/커밋
    path: k8s                                     # 저장소 내 매니페스트가 위치한 경로
  destination:
    server: https://kubernetes.default.svc        # 배포 대상 클러스터 (현재 클러스터)
    namespace: demo                               # 배포할 네임스페이스
  syncPolicy:
    automated:
      prune: true       # Git에서 삭제된 리소스를 클러스터에서도 자동 삭제
      selfHeal: true    # 클러스터 상태가 Git과 달라지면 자동으로 Git 상태로 복구
    syncOptions:
      - CreateNamespace=true    # destination.namespace가 없으면 자동 생성
```

```powershell
# application.yaml을 클러스터에 적용해 ArgoCD Application 리소스를 생성한다
kubectl apply -f application.yaml

# argocd 네임스페이스의 Application 목록과 동기화 상태를 확인한다
# SYNC STATUS: Synced / HEALTH STATUS: Healthy면 정상
kubectl -n argocd get applications

# demo 네임스페이스에 파드가 생성되었는지 확인한다
kubectl -n demo get pods
```

UI에서 `demo`가 `Synced` / `Healthy`면 성공.

## 7. GitOps 동작 확인

1. `k8s\deployment.yaml`의 `replicas`를 3으로 수정
2. push
   ```powershell
   cd C:\gitops\demo
   # 변경된 deployment.yaml을 스테이징에 추가
   git add .
   # 변경 내용을 커밋
   git commit -m "scale to 3 replicas"
   # Gitea로 push → ArgoCD가 변경을 감지하고 클러스터에 반영
   git push
   ```
3. 반영 확인
   ```powershell
   # -w(watch): 파드 목록을 실시간으로 갱신하며 보여준다
   # 파드가 3개로 늘어나는 과정을 확인할 수 있다
   kubectl -n demo get pods -w
   ```
   - 기본 폴링 주기 3분. 즉시 반영하려면 UI의 Refresh 또는 `argocd app get demo --refresh`
4. selfHeal 확인
   ```powershell
   # 클러스터의 파드를 임의로 1개로 줄인다 (Git 상태와 불일치 유발)
   kubectl -n demo scale deploy/demo-app --replicas=1

   # selfHeal=true 설정으로 ArgoCD가 Git 상태(3개)로 자동 복구하는 것을 확인한다
   kubectl -n demo get pods -w
   ```

## 8. 운영(prod) 클러스터 (나중에)

```powershell
# 'prod'라는 이름의 별도 클러스터를 생성한다
kind create cluster --name prod

# kubectl 컨텍스트를 prod 클러스터로 전환한다
kubectl config use-context kind-prod
```

ArgoCD도 prod에 별도 설치한다. Gitea는 `kind` 네트워크를 공유하므로 그대로 사용 가능. 16GB에서는 두 클러스터 동시 구동이 빠듯하니 필요할 때만 띄우고 `kind delete cluster --name prod`로 정리한다.

---

## Q&A / 트러블슈팅

### Q1. `kubectl -n argocd get pods -w` 상태에서 설치가 끝난 건가?
7개 파드(application-controller, applicationset-controller, dex-server, notifications-controller, redis, repo-server, server)가 모두 `1/1 Running`이고 RESTARTS가 0이면 완료. `-w`는 watch 모드라 커서가 멈춰 있는 것이므로 `Ctrl+C`로 빠져나온다.

### Q2. `docker build` 시 `lstat /build/libs: no such file or directory`
- 원인: jar가 아직 빌드되지 않았거나, Maven 프로젝트라 jar가 `target/`에 생성됨.
- `pom.xml` → Maven(`target/`), `build.gradle` → Gradle(`build/libs/`).
- jar를 먼저 빌드한 뒤 Dockerfile의 `COPY` 경로를 맞춘다.
- Gradle에서 `*-plain.jar`가 같이 생기면 `COPY build/libs/*.jar`가 두 파일에 매칭되어 실패한다. 실제 파일명을 직접 적는다.

### Q3. `argocd repo list` → `Argo CD server address unspecified`
- 원인: `argocd login`이 안 되어 서버 주소가 저장되지 않음.
- 해결: port-forward 창을 열어둔 상태에서 `argocd login localhost:8080 --username admin --insecure` 실행 후 다시 시도. 계속 안 되면 UI에서 저장소를 등록한다.

### Q4. 앱은 어떤 주소로 확인하나?
Service가 ClusterIP라 클러스터 밖에서 직접 접근할 수 없다. port-forward를 사용한다.

```powershell
# demo 네임스페이스의 demo-app Service를 로컬 9090 포트로 포워딩한다
# 로컬 9090 → Service 80 → 파드 8080 순으로 트래픽이 전달된다
# 8080은 ArgoCD가 이미 사용 중이므로 9090을 사용
kubectl -n demo port-forward svc/demo-app 9090:80
```

`http://localhost:9090` (8080은 ArgoCD가 사용 중). 컨트롤러가 없으면 `/`에서 Whitelabel Error Page(404)가 뜨는데, 앱이 정상 기동되어 응답하는 것이다.
접속이 안 되면:

```powershell
# 파드의 현재 상태(Running/Error/Pending 등)를 확인한다
kubectl -n demo get pods

# 파드의 로그를 출력해 애플리케이션 오류 여부를 확인한다
kubectl -n demo logs deploy/demo-app
```

### Q5. 파드가 `ErrImagePull` / `ImagePullBackOff`
`kind load docker-image demo-app:<태그> --name dev`가 안 된 상태. 다시 로드하고 `crictl images`로 확인한다.

### Q6. 저장소 등록 시 `no such host`
`docker network connect kind gitea`가 빠진 것. 실행 후 `docker network inspect kind`로 확인한다.

### Q7. 소스 수정 시 자동 반영되게 하려면?
ArgoCD는 Git → 클러스터 구간만 담당한다. 앞 구간(빌드 → 이미지 → 매니페스트 갱신)은 CI의 역할이며, 로컬 실습에서는 스크립트가 가장 가볍다.

앱 프로젝트 루트의 `deploy.ps1`

```powershell
# 현재 시각을 태그로 사용해 항상 고유한 이미지 태그를 생성한다
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$manifest = "C:\gitops\demo"

# 1. 빌드 (Gradle이면 .\gradlew bootJar)
.\mvnw clean package -DskipTests
# 빌드 실패 시 스크립트 즉시 중단 ($LASTEXITCODE: 마지막 명령의 종료 코드)
if ($LASTEXITCODE -ne 0) { exit 1 }

# 2. 이미지 빌드 + kind 로드
docker build -t demo-app:$tag .
# 새 이미지를 kind 클러스터 내부에 직접 주입
kind load docker-image demo-app:$tag --name dev

# 3. 매니페스트의 이미지 태그 교체
$f = "$manifest\k8s\deployment.yaml"
# 정규식으로 'image: demo-app:기존태그'를 새 태그로 치환한 뒤 파일에 저장
(Get-Content $f) -replace 'image: demo-app:.*', "image: demo-app:$tag" | Set-Content $f

# 4. Git push (매니페스트 변경 → Gitea push → ArgoCD 감지)
git -C $manifest add .
git -C $manifest commit -m "deploy demo-app:$tag"
git -C $manifest push

# 5. 즉시 반영 (3분 폴링 대기 없이 ArgoCD가 바로 동기화하도록 강제)
# Out-Null: 명령 출력을 버림 (스크립트 로그를 깔끔하게 유지)
argocd app get demo --refresh | Out-Null
```

```powershell
.\deploy.ps1
```

실행 정책 오류 시: `powershell -ExecutionPolicy Bypass -File .\deploy.ps1`

**태그를 타임스탬프로 하는 이유:** 같은 태그면 매니페스트가 변하지 않아 ArgoCD가 변경을 감지하지 못하고, `IfNotPresent` 때문에 새 이미지도 사용되지 않는다. 태그가 바뀌어야 Git diff → 롤링 업데이트가 일어난다.

**완전 자동(commit/push 시 자동 빌드)으로 하려면:** Gitea Actions + act_runner + 로컬 레지스트리가 필요하다. CI 러너 컨테이너에서는 `kind load`를 쓰기 어려워 이미지를 레지스트리에 push해야 하기 때문이다. 컨테이너가 2~3개 늘어 16GB PC에는 부담이므로, GitOps 흐름을 익히는 동안은 스크립트로 충분하다.
