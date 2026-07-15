# 웹 취약점 (Web Security)

## 1. OWASP란?

**OWASP (Open Web Application Security Project)**는 웹 애플리케이션 보안 향상을 위한 비영리 오픈소스 커뮤니티입니다.
매 3~4년마다 **OWASP Top 10**을 발표하여 가장 심각한 웹 취약점 목록을 제공합니다.

> 최신 버전: **OWASP Top 10 : 2021**

---

## 2. OWASP Top 10 (2021)

| 순위 | 취약점 | 설명 |
|------|--------|------|
| A01 | Broken Access Control | 접근 제어 실패 |
| A02 | Cryptographic Failures | 암호화 실패 |
| A03 | Injection | 인젝션 |
| A04 | Insecure Design | 안전하지 않은 설계 |
| A05 | Security Misconfiguration | 보안 설정 오류 |
| A06 | Vulnerable and Outdated Components | 취약하고 오래된 구성 요소 |
| A07 | Identification and Authentication Failures | 식별 및 인증 실패 |
| A08 | Software and Data Integrity Failures | 소프트웨어 및 데이터 무결성 실패 |
| A09 | Security Logging and Monitoring Failures | 보안 로깅 및 모니터링 실패 |
| A10 | Server-Side Request Forgery (SSRF) | 서버 측 요청 위조 |

---

### A01 - Broken Access Control (접근 제어 실패)

인증된 사용자가 **권한 밖의 리소스나 기능에 접근**할 수 있는 취약점으로, 2021 Top 10 중 가장 심각한 항목입니다.

**공격 예시**
```
# URL 파라미터 변조로 타인의 데이터 접근
GET /api/orders/1001   # 본인 주문
GET /api/orders/1002   # 타인 주문 → 서버가 소유자 검증 없이 반환
```

```js
// 취약한 코드 - 파라미터 ID만으로 조회
app.get("/api/orders/:id", async (req, res) => {
  const order = await Order.findById(req.params.id);
  res.json(order); // 소유자 검증 없음
});

// 안전한 코드 - 요청자의 userId와 비교 검증
app.get("/api/orders/:id", authenticate, async (req, res) => {
  const order = await Order.findById(req.params.id);
  if (!order || order.userId !== req.user.id) {
    return res.status(403).json({ error: "접근 권한 없음" });
  }
  res.json(order);
});
```

**대응 방안**
- 모든 API에서 소유자 검증 (서버 측 수행)
- 기본 정책을 "거부"로 설정, 명시적 허용만 허용
- 관리자 기능 URL 예측 불가능하게 설계
- 디렉토리 탐색 (`../`) 차단

---

### A02 - Cryptographic Failures (암호화 실패)

민감한 데이터를 암호화하지 않거나 **취약한 암호화 알고리즘**을 사용하는 취약점입니다.

**취약한 패턴**
```js
// MD5, SHA-1은 취약 - 레인보우 테이블/충돌 공격에 취약
const hash = crypto.createHash("md5").update(password).digest("hex");

// 평문 비밀번호 저장
db.users.insert({ password: "user1234" });

// HTTP로 민감 데이터 전송 (HTTPS 미사용)
fetch("http://api.example.com/login", { body: JSON.stringify({ password }) });
```

**안전한 패턴**
```js
import bcrypt from "bcrypt";

// bcrypt - 단방향 해시 + 솔트 자동 포함 (비밀번호 저장 권장)
const hashed = await bcrypt.hash(password, 12); // cost factor 12
const isMatch = await bcrypt.compare(inputPassword, hashed);

// AES-256-GCM - 양방향 암호화가 필요한 경우 (복호화 가능해야 할 때)
const algorithm = "aes-256-gcm";
const key = crypto.randomBytes(32);
const iv = crypto.randomBytes(16);
const cipher = crypto.createCipheriv(algorithm, key, iv);
```

**대응 방안**
- 비밀번호: **bcrypt / Argon2 / scrypt** 사용 (MD5·SHA-1 절대 금지)
- 전송 데이터: **HTTPS(TLS 1.2+)** 필수
- 민감 데이터(카드 번호 등): AES-256 이상 암호화
- 암호화 키는 환경 변수로 분리, 소스코드에 하드코딩 금지

---

### A03 - Injection (인젝션)

사용자 입력값이 **SQL, OS 명령어, LDAP 쿼리 등에 그대로 삽입**되어 의도치 않은 명령이 실행되는 취약점입니다.

#### SQL Injection

```sql
-- 취약한 쿼리 - 입력값 그대로 문자열 연결
SELECT * FROM users WHERE id = '입력값'

-- 공격 입력: ' OR '1'='1
SELECT * FROM users WHERE id = '' OR '1'='1'  -- 전체 데이터 노출
-- 공격 입력: '; DROP TABLE users; --
SELECT * FROM users WHERE id = ''; DROP TABLE users; --'
```

```js
// 취약한 코드
const query = `SELECT * FROM users WHERE email = '${email}'`;

// 안전한 코드 - Prepared Statement (파라미터 바인딩)
const query = "SELECT * FROM users WHERE email = ?";
db.query(query, [email]);

// ORM 사용 (Sequelize, Prisma 등)
const user = await User.findOne({ where: { email } }); // 자동으로 파라미터 바인딩
```

#### Command Injection

```js
// 취약한 코드 - 사용자 입력을 OS 명령어에 직접 삽입
const { exec } = require("child_process");
exec(`ping ${req.query.host}`, callback);
// 공격: host=127.0.0.1; rm -rf /

// 안전한 코드 - 인자를 배열로 분리 전달
const { execFile } = require("child_process");
execFile("ping", [req.query.host], callback);

// 또는 입력값 화이트리스트 검증 후 사용
const validHost = /^[\w.-]+$/.test(host);
if (!validHost) return res.status(400).json({ error: "유효하지 않은 입력" });
```

#### XSS (Cross-Site Scripting) - Injection의 일종

사용자 입력이 HTML에 삽입되어 악성 스크립트가 실행됩니다.

```js
// 취약한 코드 - 입력값을 HTML에 그대로 삽입
div.innerHTML = userInput;
// 공격: <script>document.cookie를 공격자 서버로 전송</script>

// 안전한 코드 - innerText 사용 또는 입력값 이스케이프
div.textContent = userInput; // 스크립트 실행 안 됨

// 서버 측 이스케이프 (Node.js)
import escapeHtml from "escape-html";
const safe = escapeHtml(userInput);

// React는 기본적으로 JSX 내 값을 자동 이스케이프
<div>{userInput}</div>  // 안전
<div dangerouslySetInnerHTML={{ __html: userInput }} />  // 위험! 직접 삽입 금지
```

**대응 방안**
- SQL: Prepared Statement 또는 ORM 사용
- OS 명령어: 사용자 입력을 명령어에 직접 사용 금지
- XSS: 출력 시 이스케이프, CSP 헤더 적용

---

### A04 - Insecure Design (안전하지 않은 설계)

구현 단계의 버그가 아닌 **설계·아키텍처 단계의 보안 결함**입니다.

**취약한 설계 예시**
```
- 비밀번호 재설정 시 힌트 질문만으로 인증 (무차별 대입 공격 가능)
- 결제 금액을 클라이언트에서 결정하고 서버로 전달 (변조 가능)
- 단일 API 키로 모든 서비스 접근 (키 탈취 시 전체 노출)
```

**대응 방안**
- 설계 단계부터 위협 모델링 (Threat Modeling) 수행
- 비즈니스 로직 검증은 반드시 서버에서 수행
- 금액·권한 등 중요 데이터는 클라이언트 입력값을 신뢰하지 않음
- 최소 권한 원칙 (Principle of Least Privilege) 적용

---

### A05 - Security Misconfiguration (보안 설정 오류)

서버·프레임워크·데이터베이스 등의 **잘못된 보안 설정**으로 발생하는 취약점입니다.

**취약한 설정 예시**
```bash
# 불필요한 서비스/포트 노출
# 기본 계정/비밀번호 사용 (admin/admin)
# 에러 메시지에 스택 트레이스·DB 정보 노출
# 디렉토리 목록(Directory Listing) 활성화
# CORS를 모든 출처에 허용
Access-Control-Allow-Origin: *
```

**안전한 설정**
```js
// Express - 보안 헤더 설정 (helmet)
import helmet from "helmet";
app.use(helmet());

// CORS - 허용 출처 명시
app.use(cors({ origin: ["https://my-app.com"] }));

// 에러 메시지 - 프로덕션에서 상세 오류 숨김
app.use((err, req, res, next) => {
  if (process.env.NODE_ENV === "production") {
    res.status(500).json({ error: "서버 오류" }); // 스택 미노출
  } else {
    res.status(500).json({ error: err.message, stack: err.stack });
  }
});
```

```yaml
# nginx - 서버 버전 정보 숨김
server_tokens off;

# 불필요한 HTTP 메서드 차단
if ($request_method !~ ^(GET|POST|PUT|DELETE)$) {
  return 405;
}
```

---

### A06 - Vulnerable and Outdated Components (취약한 구성 요소)

**알려진 취약점이 있는 오래된 라이브러리·프레임워크**를 사용하는 경우입니다.

**대응 방안**
```bash
# 취약점 스캔
npm audit                      # 취약한 패키지 목록 확인
npm audit fix                  # 자동 업데이트 가능한 항목 수정
npm audit fix --force          # 하위 호환성 무시하고 강제 업데이트

# 의존성 현황 확인
npm outdated                   # 업데이트 가능한 패키지 목록

# Snyk - 더 정밀한 취약점 스캔 도구
npx snyk test
```

```yaml
# GitHub Actions - CI에서 자동 취약점 스캔
- name: Security Audit
  run: npm audit --audit-level=high  # high 이상 취약점 발견 시 빌드 실패
```

---

### A07 - Identification and Authentication Failures (인증 실패)

**인증·세션 관리의 취약한 구현**으로 발생합니다.

**취약한 패턴**
```
- 무제한 로그인 시도 허용 (무차별 대입 공격)
- 약한 JWT 시크릿 키 사용
- 만료되지 않는 세션 토큰
- 비밀번호 재설정 URL 예측 가능
- 다중 인증(MFA) 미적용
```

**안전한 구현**
```js
import rateLimit from "express-rate-limit";
import jwt from "jsonwebtoken";

// 로그인 Rate Limiting - IP당 5분에 5회로 제한
const loginLimiter = rateLimit({
  windowMs: 5 * 60 * 1000,
  max: 5,
  message: "로그인 시도 횟수 초과. 5분 후 다시 시도하세요.",
});
app.post("/api/login", loginLimiter, loginHandler);

// JWT - 충분히 긴 시크릿 키 + 짧은 만료 시간
const token = jwt.sign(
  { userId: user.id, role: user.role },
  process.env.JWT_SECRET,  // 최소 32자 이상 랜덤 문자열
  { expiresIn: "1h" }      // 짧은 만료 (Refresh Token 패턴 병행)
);

// 비밀번호 재설정 - 예측 불가능한 토큰 사용
const resetToken = crypto.randomBytes(32).toString("hex");
const hashedToken = crypto.createHash("sha256").update(resetToken).digest("hex");
// DB에는 hashedToken 저장, 이메일에는 resetToken 전송
```

---

### A08 - Software and Data Integrity Failures (무결성 실패)

**검증 없이 외부 소스의 코드나 데이터를 신뢰**할 때 발생합니다. CI/CD 파이프라인 공격, 악성 npm 패키지 삽입 등이 포함됩니다.

**취약한 패턴**
```html
<!-- SRI(Subresource Integrity) 없이 CDN 스크립트 로드 -->
<script src="https://cdn.example.com/jquery.min.js"></script>
<!-- CDN이 해킹되면 악성 스크립트 실행 가능 -->
```

**안전한 패턴**
```html
<!-- SRI 해시로 파일 무결성 검증 -->
<script
  src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"
  integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo="
  crossorigin="anonymous"
></script>
```

```json
// package-lock.json / npm-shrinkwrap.json 반드시 커밋
// 패키지 버전과 해시를 고정하여 의도치 않은 버전 변경 방지
{
  "node_modules/lodash": {
    "version": "4.17.21",
    "resolved": "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz",
    "integrity": "sha512-v2kDEe57lecTulaDIuNTPy3Ry4gLGJ6Z1O3vE1krgXZNrsQ+LFTGHVxVjcXPs17LhbZa2e25yyBhfDn0BLSL=="
  }
}
```

---

### A09 - Security Logging and Monitoring Failures (로깅 실패)

**보안 이벤트를 로깅하지 않거나 모니터링하지 않아** 공격을 탐지하지 못하는 취약점입니다.

**반드시 기록해야 할 이벤트**
```
- 로그인 성공/실패 (IP, 시각, 사용자)
- 권한 없는 접근 시도 (403 응답)
- 입력값 검증 실패 (SQL 인젝션 패턴 탐지 등)
- 관리자 기능 사용
- 계정 잠금, 비밀번호 변경
```

**안전한 로깅 구현**
```js
import winston from "winston";

const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: "error.log", level: "error" }),
    new winston.transports.File({ filename: "security.log" }),
  ],
});

// 보안 이벤트 기록
app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  const user = await findUser(email);

  if (!user || !(await bcrypt.compare(password, user.password))) {
    logger.warn("로그인 실패", {
      email,
      ip: req.ip,
      userAgent: req.headers["user-agent"],
    });
    return res.status(401).json({ error: "인증 실패" });
  }

  logger.info("로그인 성공", { userId: user.id, ip: req.ip });
  // ...
});
```

> 로그에 **비밀번호·카드번호 등 민감 정보를 절대 기록하지 않습니다.**

---

### A10 - Server-Side Request Forgery, SSRF (서버 측 요청 위조)

**서버가 공격자가 제공한 URL로 내부 요청**을 보내도록 유도하는 취약점입니다. 클라우드 환경의 메타데이터 서버(169.254.169.254)나 내부 네트워크에 접근 가능합니다.

**공격 예시**
```
# 이미지 URL을 입력받아 서버가 fetch하는 기능 악용
POST /api/fetch-image
{ "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials" }
# AWS 내부 메타데이터(자격 증명)가 응답으로 반환됨

{ "url": "http://localhost:6379" }  # 내부 Redis에 접근
{ "url": "file:///etc/passwd" }     # 서버 파일 읽기 시도
```

**대응 방안**
```js
import { URL } from "url";

function isSafeUrl(urlStr) {
  try {
    const url = new URL(urlStr);

    // 허용된 스킴만 허용
    if (!["http:", "https:"].includes(url.protocol)) return false;

    // 내부 IP 대역 차단
    const blockedHosts = ["localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"];
    if (blockedHosts.includes(url.hostname)) return false;

    // 사설 IP 대역 차단 (정규식)
    const privateIp = /^(10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)/;
    if (privateIp.test(url.hostname)) return false;

    return true;
  } catch {
    return false;
  }
}

app.post("/api/fetch-image", async (req, res) => {
  if (!isSafeUrl(req.body.url)) {
    return res.status(400).json({ error: "허용되지 않는 URL" });
  }
  const response = await fetch(req.body.url);
  // ...
});
```

---

## 3. OWASP 외 주요 웹 취약점

### CSRF (Cross-Site Request Forgery)

**인증된 사용자를 속여** 의도하지 않은 요청을 서버에 전송하게 만드는 공격입니다.

```html
<!-- 공격자 사이트에 숨겨진 폼 - 피해자가 방문하면 자동 제출 -->
<img src="http://bank.com/transfer?to=attacker&amount=1000000" />
<form action="http://bank.com/transfer" method="POST">
  <input type="hidden" name="to" value="attacker" />
  <input type="hidden" name="amount" value="1000000" />
</form>
<script>document.forms[0].submit();</script>
```

**대응 방안**
```js
import csrf from "csurf";

// 1. CSRF 토큰 - 폼마다 서버 발급 토큰 포함 후 검증
app.use(csrf({ cookie: true }));
app.get("/form", (req, res) => {
  res.render("form", { csrfToken: req.csrfToken() });
});

// 2. SameSite 쿠키 속성 설정
res.cookie("session", token, { sameSite: "strict" });

// 3. Referer / Origin 헤더 검증
app.use((req, res, next) => {
  const origin = req.headers.origin || req.headers.referer;
  if (req.method !== "GET" && !origin?.startsWith("https://my-app.com")) {
    return res.status(403).json({ error: "CSRF 차단" });
  }
  next();
});
```

---

### Clickjacking (클릭재킹)

**투명한 iframe으로 정상 사이트를 덮어** 사용자가 의도치 않은 버튼을 클릭하게 만드는 공격입니다.

```js
// X-Frame-Options 헤더 - iframe 내 삽입 차단
res.setHeader("X-Frame-Options", "DENY");        // 모든 iframe 차단
res.setHeader("X-Frame-Options", "SAMEORIGIN");  // 같은 출처만 허용

// CSP frame-ancestors - 더 세밀한 제어 (권장)
res.setHeader("Content-Security-Policy", "frame-ancestors 'none'");

// helmet 사용 시 (Express)
app.use(helmet.frameguard({ action: "deny" }));
```

---

### 보안 HTTP 헤더

```js
// Express + helmet으로 주요 보안 헤더 일괄 적용
import helmet from "helmet";

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'nonce-{random}'"],  // 인라인 스크립트 nonce 방식
      styleSrc: ["'self'", "https://fonts.googleapis.com"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https://api.my-app.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      objectSrc: ["'none'"],
      upgradeInsecureRequests: [],
    },
  },
  hsts: { maxAge: 31536000, includeSubDomains: true, preload: true },
}));
```

| 헤더 | 설명 |
|------|------|
| `Content-Security-Policy` (CSP) | 허용된 리소스 출처만 로드 (XSS 방어 핵심) |
| `X-Frame-Options` | iframe 삽입 제한 (Clickjacking 방어) |
| `X-Content-Type-Options: nosniff` | MIME 타입 스니핑 방지 |
| `Strict-Transport-Security` (HSTS) | HTTPS 강제 |
| `Referrer-Policy` | Referer 헤더 노출 범위 제한 |
| `Permissions-Policy` | 카메라·마이크 등 브라우저 기능 제한 |

---

### 디렉토리 탐색 (Path Traversal)

```js
// 취약한 코드 - 사용자 입력을 파일 경로에 직접 사용
app.get("/file", (req, res) => {
  const filename = req.query.name;
  res.sendFile(`/var/www/files/${filename}`);
  // 공격: ?name=../../../../etc/passwd
});

// 안전한 코드 - path.basename으로 경로 조작 차단
import path from "path";

app.get("/file", (req, res) => {
  const filename = path.basename(req.query.name); // 경로 구분자 제거
  const filePath = path.join("/var/www/files", filename);

  // 허용된 디렉토리 내 파일인지 확인 (추가 검증)
  if (!filePath.startsWith("/var/www/files")) {
    return res.status(403).json({ error: "접근 불가" });
  }
  res.sendFile(filePath);
});
```

---

## 4. 입력값 검증 (Input Validation)

```js
import Joi from "joi";
import { body, validationResult } from "express-validator";

// Joi - 스키마 기반 검증
const schema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().min(8).max(128).required(),
  age: Joi.number().integer().min(1).max(150),
});

app.post("/api/register", async (req, res) => {
  const { error } = schema.validate(req.body);
  if (error) return res.status(400).json({ error: error.details[0].message });
  // ...
});

// express-validator
app.post(
  "/api/login",
  body("email").isEmail().normalizeEmail(),
  body("password").isLength({ min: 8 }).trim(),
  (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) return res.status(400).json({ errors: errors.array() });
  }
);
```

---

## 5. 보안 점검 도구

| 도구 | 종류 | 설명 |
|------|------|------|
| **OWASP ZAP** | DAST | 실행 중인 웹앱 자동 취약점 스캔 |
| **Burp Suite** | DAST | 수동·자동 웹 취약점 점검 (프록시) |
| **SonarQube** | SAST | 소스코드 정적 분석 |
| **npm audit** | SCA | npm 패키지 취약점 검사 |
| **Snyk** | SCA | 코드·패키지·컨테이너 취약점 스캔 |
| **Trivy** | SCA | 컨테이너 이미지 취약점 스캔 |
| **Nikto** | DAST | 웹 서버 설정 취약점 스캔 |

```bash
# OWASP ZAP - Docker로 빠르게 스캔
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t https://my-app.com -r zap-report.html

# Trivy - Docker 이미지 취약점 스캔
trivy image my-app:latest

# Snyk
npx snyk test               # 패키지 취약점
npx snyk code test          # 소스코드 정적 분석
```

---

## 6. 보안 체크리스트

### 인증/인가
- [ ] 모든 API에 인증 적용 (공개 API는 명시적으로 예외 처리)
- [ ] JWT 시크릿 키 32자 이상, 환경 변수 관리
- [ ] 토큰 만료 시간 설정 (Access: 1시간, Refresh: 7~30일)
- [ ] 로그인 Rate Limiting 적용 (IP당 5분에 5회 등)
- [ ] 비밀번호 bcrypt(cost 12 이상) 해싱

### 데이터 보호
- [ ] 비밀번호·카드번호 등 민감 데이터 암호화 저장
- [ ] HTTPS 강제 (HSTS 적용)
- [ ] 로그에 민감 데이터 제외

### 입력값/출력값
- [ ] 모든 사용자 입력값 서버 측 검증
- [ ] SQL Prepared Statement 또는 ORM 사용
- [ ] HTML 출력 시 이스케이프 처리
- [ ] 파일 업로드 - 확장자·MIME 타입·크기 검증

### 헤더/설정
- [ ] 보안 헤더 설정 (helmet 등)
- [ ] CORS 허용 출처 명시적 설정
- [ ] 에러 메시지에 스택 트레이스 미노출 (프로덕션)
- [ ] 불필요한 서버 버전 정보 노출 차단

### 의존성/운영
- [ ] npm audit 정기 실행 + CI 연동
- [ ] 패키지 lock 파일 커밋
- [ ] 보안 이벤트 로깅 및 모니터링
- [ ] 정기적인 취약점 스캔 (OWASP ZAP, Snyk 등)
