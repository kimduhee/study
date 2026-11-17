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

```powershell
wsl --shutdown    # 이후 Docker Desktop 재시작
```

Docker Desktop은 WSL2 백엔드여야 한다.

## 1. CLI 설치

```powershell
winget install Kubernetes.kubectl
winget install Kubernetes.kind
winget install argoproj.argocd    # 안 되면 GitHub releases에서 argocd-windows-amd64.exe 다운로드
```

## 2. kind 클러스터 생성

```powershell
kind create cluster --name dev
kubectl config use-context kind-dev
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
      - GITEA__database__DB_TYPE=sqlite3
      - GITEA__server__ROOT_URL=http://localhost:3000/
    ports:
      - "3000:3000"
    volumes:
      - ./data:/data
    restart: unless-stopped
```

```powershell
cd C:\gitops\gitea
docker compose up -d
docker network connect kind gitea
```

초기 설정:
1. `http://localhost:3000` 접속
2. 기본값 유지, 맨 아래 관리자 계정(아이디/비번/이메일) 입력 후 설치
3. 로그인 후 `+` → 새 저장소 → 이름 `demo`

확인:

```powershell
docker ps
docker network inspect kind    # Containers에 gitea, dev-control-plane 둘 다 있어야 함
```

> ArgoCD(클러스터 내부)에서 `localhost`는 파드 자신을 가리킨다. 그래서 이후 저장소 주소는 컨테이너명 `http://gitea:3000`을 쓴다.

## 4. ArgoCD 설치

```powershell
kubectl create namespace argocd
kubectl apply -n argocd --server-side -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl -n argocd get pods -w    # 7개 파드 모두 1/1 Running 이면 완료 (Ctrl+C로 종료)
```

UI 접속 (이 창은 계속 열어둔다):

```powershell
kubectl -n argocd port-forward svc/argocd-server 8080:443
```

새 PowerShell 창에서 초기 비밀번호 확인:

```powershell
argocd admin initial-password -n argocd
# 또는
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | % { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }
```

`https://localhost:8080` (인증서 경고 무시), 계정 `admin`.

## 5. 앱 이미지 + 매니페스트 준비

### A. jar 빌드 + 이미지 빌드 + kind 로드

```powershell
.\mvnw clean package -DskipTests    # Gradle: .\gradlew bootJar
```

`Dockerfile` (프로젝트 루트)

```dockerfile
FROM eclipse-temurin:21-jre
COPY target/*.jar app.jar          # Gradle: build/libs/*.jar
EXPOSE 8080
ENTRYPOINT ["java","-jar","/app.jar"]
```

```powershell
docker build -t demo-app:0.1 .
kind load docker-image demo-app:0.1 --name dev
docker exec -it dev-control-plane crictl images | findstr demo-app
```

- 레지스트리 없이 `kind load`로 클러스터에 직접 이미지를 넣는다.
- 태그를 `latest`로 쓰면 안 된다 (항상 pull을 시도해서 실패).

### B. 매니페스트 작성 (앱 소스와 별도 폴더)

```powershell
mkdir C:\gitops\demo\k8s
cd C:\gitops\demo
```

`k8s\deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: demo-app
  template:
    metadata:
      labels:
        app: demo-app
    spec:
      containers:
        - name: demo-app
          image: demo-app:0.1
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
```

`k8s\service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: demo-app
spec:
  selector:
    app: demo-app
  ports:
    - port: 80
      targetPort: 8080
```

앱 포트가 8080이 아니면 `containerPort`, `targetPort`를 맞춘다.

### C. Gitea에 push

```powershell
cd C:\gitops\demo
git init -b main
git add .
git commit -m "init manifests"
git remote add origin http://localhost:3000/<유저>/demo.git
git push -u origin main
```

## 6. ArgoCD에 저장소 + Application 등록

### 로그인 (CLI 사용 전 필수)

port-forward 창이 열려 있는 상태에서:

```powershell
argocd login localhost:8080 --username admin --insecure
```

### 저장소 등록

```powershell
argocd repo add http://gitea:3000/<유저>/demo.git --username <유저> --password <비번>
argocd repo list    # STATUS: Successful
```

CLI가 막히면 UI: Settings → Repositories → Connect Repo (VIA HTTPS / git / URL 입력 / Gitea 계정).

### Application 생성

`C:\gitops\demo\application.yaml` — `k8s` 폴더 **밖**에 둔다 (안에 두면 ArgoCD가 자기 자신을 배포 대상으로 읽는다).

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: demo
  namespace: argocd
spec:
  project: default
  source:
    repoURL: http://gitea:3000/<유저>/demo.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: demo
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

```powershell
kubectl apply -f application.yaml
kubectl -n argocd get applications
kubectl -n demo get pods
```

UI에서 `demo`가 `Synced` / `Healthy`면 성공.

## 7. GitOps 동작 확인

1. `k8s\deployment.yaml`의 `replicas`를 3으로 수정
2. push
   ```powershell
   cd C:\gitops\demo
   git add .
   git commit -m "scale to 3 replicas"
   git push
   ```
3. 반영 확인
   ```powershell
   kubectl -n demo get pods -w
   ```
   - 기본 폴링 주기 3분. 즉시 반영하려면 UI의 Refresh 또는 `argocd app get demo --refresh`
4. selfHeal 확인
   ```powershell
   kubectl -n demo scale deploy/demo-app --replicas=1
   kubectl -n demo get pods -w    # ArgoCD가 Git 상태(3개)로 되돌림
   ```

## 8. 운영(prod) 클러스터 (나중에)

```powershell
kind create cluster --name prod
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
kubectl -n demo port-forward svc/demo-app 9090:80
```

`http://localhost:9090` (8080은 ArgoCD가 사용 중). 컨트롤러가 없으면 `/`에서 Whitelabel Error Page(404)가 뜨는데, 앱이 정상 기동되어 응답하는 것이다.
접속이 안 되면:

```powershell
kubectl -n demo get pods
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
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$manifest = "C:\gitops\demo"

# 1. 빌드 (Gradle이면 .\gradlew bootJar)
.\mvnw clean package -DskipTests
if ($LASTEXITCODE -ne 0) { exit 1 }

# 2. 이미지 빌드 + kind 로드
docker build -t demo-app:$tag .
kind load docker-image demo-app:$tag --name dev

# 3. 매니페스트의 이미지 태그 교체
$f = "$manifest\k8s\deployment.yaml"
(Get-Content $f) -replace 'image: demo-app:.*', "image: demo-app:$tag" | Set-Content $f

# 4. Git push
git -C $manifest add .
git -C $manifest commit -m "deploy demo-app:$tag"
git -C $manifest push

# 5. 즉시 반영
argocd app get demo --refresh | Out-Null
```

```powershell
.\deploy.ps1
```

실행 정책 오류 시: `powershell -ExecutionPolicy Bypass -File .\deploy.ps1`

**태그를 타임스탬프로 하는 이유:** 같은 태그면 매니페스트가 변하지 않아 ArgoCD가 변경을 감지하지 못하고, `IfNotPresent` 때문에 새 이미지도 사용되지 않는다. 태그가 바뀌어야 Git diff → 롤링 업데이트가 일어난다.

**완전 자동(commit/push 시 자동 빌드)으로 하려면:** Gitea Actions + act_runner + 로컬 레지스트리가 필요하다. CI 러너 컨테이너에서는 `kind load`를 쓰기 어려워 이미지를 레지스트리에 push해야 하기 때문이다. 컨테이너가 2~3개 늘어 16GB PC에는 부담이므로, GitOps 흐름을 익히는 동안은 스크립트로 충분하다.
