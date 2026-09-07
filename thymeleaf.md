# Thymeleaf

## 1. 개요 (Overview)

Thymeleaf는 **Spring Boot와 공식 통합되는 Java 서버사이드 템플릿 엔진**입니다.
HTML 파일 그대로 브라우저에서 열어볼 수 있는 **Natural Template** 방식을 지원하며,
서버를 거치면 `th:*` 속성이 실제 값으로 치환되어 렌더링됩니다.

| 구분 | 설명 |
|------|------|
| 템플릿 방식 | HTML 속성 기반 (`th:*`) |
| Spring Boot 통합 | 자동 설정(AutoConfiguration) 지원 |
| 기본 경로 | `src/main/resources/templates/` |
| 파일 확장자 | `.html` |

---

## 2. 설정 (Setup)

### 의존성 추가

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-thymeleaf</artifactId>
</dependency>
```

```gradle
// build.gradle
implementation 'org.springframework.boot:spring-boot-starter-thymeleaf'
```

### application.yml 설정

```yaml
spring:
  thymeleaf:
    prefix: classpath:/templates/   # 템플릿 기본 경로 (기본값)
    suffix: .html                   # 파일 확장자 (기본값)
    encoding: UTF-8
    mode: HTML
    cache: false                    # 개발 시 캐시 비활성화 (운영에서는 true)
```

### 프로젝트 구조

```
src/main/resources/
├─ templates/
│   ├─ index.html          # 메인 페이지
│   ├─ layout/
│   │   └─ base.html       # 공통 레이아웃 (fragment)
│   └─ user/
│       ├─ list.html
│       └─ detail.html
└─ static/
    ├─ css/
    ├─ js/
    └─ images/
```

### HTML 네임스페이스 선언

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org"
      xmlns:sec="http://www.thymeleaf.org/extras/spring-security">
<head>
    <meta charset="UTF-8">
    <title>Thymeleaf 예제</title>
</head>
<body>
    ...
</body>
</html>
```

### Controller 기본 구조

```java
@Controller
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/users")
    public String list(Model model) {
        model.addAttribute("users", userService.findAll());
        model.addAttribute("title", "사용자 목록");
        return "user/list";  // templates/user/list.html
    }

    @GetMapping("/users/{id}")
    public String detail(@PathVariable Long id, Model model) {
        model.addAttribute("user", userService.findById(id));
        return "user/detail";
    }
}
```

---

## 3. 표현식 (Expressions)

| 표현식 | 문법 | 설명 |
|--------|------|------|
| 변수 표현식 | `${...}` | Model에 담긴 변수 접근 |
| 선택 변수 표현식 | `*{...}` | `th:object`로 선택된 객체의 필드 접근 |
| 메시지 표현식 | `#{...}` | 메시지 파일(i18n) 조회 |
| URL 표현식 | `@{...}` | URL 생성 |
| 조각 표현식 | `~{...}` | 템플릿 조각(fragment) 참조 |

```html
<!-- 변수 표현식 - Model의 값 출력 -->
<p th:text="${user.name}">기본 이름</p>

<!-- 선택 변수 표현식 - th:object로 user 선택 후 필드 바로 접근 -->
<div th:object="${user}">
    <p th:text="*{name}">이름</p>
    <p th:text="*{email}">이메일</p>
</div>

<!-- 메시지 표현식 - messages.properties의 키 -->
<p th:text="#{welcome.message}">환영합니다</p>

<!-- URL 표현식 -->
<a th:href="@{/users}">목록</a>
<a th:href="@{/users/{id}(id=${user.id})}">상세</a>
<a th:href="@{/search(keyword=${keyword},page=1)}">검색</a>
```

---

## 4. 기본 속성 문법

### 텍스트 출력

```html
<!-- th:text - 텍스트로 출력 (HTML 이스케이프 적용) -->
<p th:text="${message}">기본 텍스트</p>

<!-- th:utext - HTML 태그 포함 그대로 출력 (XSS 주의) -->
<p th:utext="${htmlContent}">기본 내용</p>

<!-- 인라인 표현식 - 태그 내부에 텍스트와 변수 혼용 -->
<p>안녕하세요, [[${user.name}]]님!</p>       <!-- 이스케이프 적용 -->
<p>HTML: [(${htmlContent})]</p              <!-- 이스케이프 미적용 -->
```

### 속성 바인딩

```html
<!-- th:attr - 임의 속성 설정 -->
<input th:attr="placeholder=${placeholder}" />

<!-- 속성별 전용 th:속성명 (권장) -->
<input th:value="${user.name}" />
<input th:placeholder="${hint}" />
<img th:src="@{/images/logo.png}" th:alt="${altText}" />
<a th:href="@{/users}" th:title="${linkTitle}">링크</a>

<!-- 클래스 바인딩 -->
<div th:class="${isActive} ? 'active' : 'inactive'">상태</div>
<div th:classappend="${isError} ? 'error'">내용</div>   <!-- 기존 class에 추가 -->

<!-- 스타일 바인딩 -->
<div th:style="'color: ' + ${color} + '; font-size: 14px'">텍스트</div>

<!-- th:id, th:name -->
<input th:id="'user-' + ${user.id}" th:name="${fieldName}" />
```

### 조건부 속성 (boolean)

```html
<!-- checked, selected, disabled, readonly 등 boolean 속성 -->
<input type="checkbox" th:checked="${user.active}" />
<option th:selected="${item.id == selectedId}" th:value="${item.id}">옵션</option>
<button th:disabled="${!isAdmin}">삭제</button>
<input type="text" th:readonly="${isReadOnly}" />
```

---

## 5. 조건부 렌더링

### th:if / th:unless

```html
<!-- th:if - 조건이 true일 때 렌더링 -->
<p th:if="${user.admin}">관리자입니다.</p>

<!-- th:unless - 조건이 false일 때 렌더링 (th:if의 반대) -->
<p th:unless="${user.admin}">일반 사용자입니다.</p>

<!-- null 또는 빈 값 체크 -->
<p th:if="${message != null and !#strings.isEmpty(message)}" th:text="${message}"></p>

<!-- Elvis 연산자 - null이면 기본값 -->
<p th:text="${user.nickname} ?: '닉네임 없음'">닉네임</p>
```

### th:switch / th:case

```html
<div th:switch="${user.role}">
    <p th:case="'ADMIN'">관리자</p>
    <p th:case="'MANAGER'">매니저</p>
    <p th:case="*">일반 사용자</p>   <!-- default -->
</div>
```

---

## 6. 반복문 (th:each)

```html
<!-- 기본 반복 -->
<ul>
    <li th:each="user : ${users}" th:text="${user.name}">이름</li>
</ul>

<!-- iterStat 상태 변수 (선택적) -->
<tr th:each="user, stat : ${users}">
    <td th:text="${stat.index}">0</td>      <!-- 0부터 시작 인덱스 -->
    <td th:text="${stat.count}">1</td>      <!-- 1부터 시작 카운트 -->
    <td th:text="${stat.size}">10</td>      <!-- 전체 크기 -->
    <td th:text="${user.name}">이름</td>
    <td th:text="${stat.first} ? '첫번째'">첫번째 여부</td>
    <td th:text="${stat.last} ? '마지막'">마지막 여부</td>
    <td th:class="${stat.odd} ? 'odd' : 'even'">홀짝</td>
</tr>

<!-- Map 반복 -->
<div th:each="entry : ${map}">
    <span th:text="${entry.key}">키</span>:
    <span th:text="${entry.value}">값</span>
</div>

<!-- 숫자 범위 반복 -->
<li th:each="i : ${#numbers.sequence(1, 10)}" th:text="${i}">1</li>
```

---

## 7. URL 표현식 (@{...})

```html
<!-- 정적 URL -->
<a th:href="@{/users}">목록</a>

<!-- 경로 변수 -->
<a th:href="@{/users/{id}(id=${user.id})}">상세</a>
<!-- 결과: /users/1 -->

<!-- 쿼리 파라미터 -->
<a th:href="@{/search(keyword=${keyword},page=${page})}">검색</a>
<!-- 결과: /search?keyword=홍길동&page=1 -->

<!-- 경로 변수 + 쿼리 파라미터 혼합 -->
<a th:href="@{/users/{id}/posts(id=${user.id},page=${page})}">게시물</a>
<!-- 결과: /users/1/posts?page=1 -->

<!-- 절대 URL -->
<a th:href="@{https://www.example.com}">외부 링크</a>

<!-- 정적 리소스 -->
<link th:href="@{/css/style.css}" rel="stylesheet" />
<script th:src="@{/js/app.js}"></script>
<img th:src="@{/images/{name}(name=${image.fileName})}" />
```

---

## 8. 폼 처리 (Form)

```java
// Controller
@GetMapping("/users/new")
public String createForm(Model model) {
    model.addAttribute("userForm", new UserForm());
    return "user/form";
}

@PostMapping("/users")
public String create(@Valid @ModelAttribute("userForm") UserForm form,
                     BindingResult result) {
    if (result.hasErrors()) {
        return "user/form";
    }
    userService.save(form);
    return "redirect:/users";
}
```

```html
<!-- 폼 - th:action, th:object -->
<form th:action="@{/users}" th:object="${userForm}" method="post">

    <!-- th:field - name, id, value 자동 설정 -->
    <div>
        <label for="name">이름</label>
        <input type="text" th:field="*{name}" />
        <!-- 결과: id="name" name="name" value="홍길동" -->

        <!-- 검증 오류 메시지 -->
        <span th:if="${#fields.hasErrors('name')}"
              th:errors="*{name}"
              class="error">이름 오류</span>
    </div>

    <div>
        <label for="email">이메일</label>
        <input type="email" th:field="*{email}" />
        <span th:errors="*{email}" class="error"></span>
    </div>

    <!-- select -->
    <select th:field="*{role}">
        <option value="">선택하세요</option>
        <option th:each="role : ${roles}"
                th:value="${role.code}"
                th:text="${role.name}">역할</option>
    </select>

    <!-- checkbox - 단일 -->
    <input type="checkbox" th:field="*{active}" />

    <!-- checkbox - 복수 -->
    <input type="checkbox" th:each="hobby : ${allHobbies}"
           th:field="*{hobbies}"
           th:value="${hobby.id}"
           th:text="${hobby.name}" />

    <!-- radio -->
    <input type="radio" th:each="gender : ${genders}"
           th:field="*{gender}"
           th:value="${gender.code}"
           th:text="${gender.name}" />

    <!-- textarea -->
    <textarea th:field="*{content}"></textarea>

    <!-- CSRF 토큰 (Spring Security 사용 시 자동 포함) -->
    <input type="hidden" th:name="${_csrf.parameterName}" th:value="${_csrf.token}" />

    <button type="submit">저장</button>
</form>
```

### 폼 검증 오류 전체 표시

```html
<!-- 전체 오류 목록 -->
<div th:if="${#fields.hasAnyErrors()}">
    <ul>
        <li th:each="err : ${#fields.allErrors()}" th:text="${err}">오류</li>
    </ul>
</div>

<!-- global 오류 (특정 필드가 아닌 오브젝트 단위 오류) -->
<p th:if="${#fields.hasGlobalErrors()}"
   th:each="err : ${#fields.globalErrors()}"
   th:text="${err}">글로벌 오류</p>
```

---

## 9. 레이아웃 (Fragment)

### Fragment 정의

```html
<!-- templates/layout/base.html -->
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head th:fragment="head(title)">
    <meta charset="UTF-8">
    <title th:text="${title} + ' - My App'">My App</title>
    <link th:href="@{/css/style.css}" rel="stylesheet" />
</head>
<body>

<!-- 헤더 fragment -->
<header th:fragment="header">
    <nav>
        <a th:href="@{/}">홈</a>
        <a th:href="@{/users}">사용자</a>
    </nav>
</header>

<!-- 푸터 fragment -->
<footer th:fragment="footer">
    <p>© 2025 My App</p>
</footer>

</body>
</html>
```

### Fragment 사용

```html
<!-- templates/user/list.html -->
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">

<!-- th:replace - 현재 태그를 fragment로 완전 교체 (권장) -->
<head th:replace="~{layout/base :: head('사용자 목록')}"></head>

<body>
    <header th:replace="~{layout/base :: header}"></header>

    <!-- th:insert - 현재 태그 안에 fragment 삽입 -->
    <div th:insert="~{layout/base :: header}"></div>

    <main>
        <h1>사용자 목록</h1>
        <!-- 내용 -->
    </main>

    <footer th:replace="~{layout/base :: footer}"></footer>
</body>
</html>
```

### 파라미터가 있는 Fragment

```html
<!-- fragment 정의 - 파라미터 수신 -->
<div th:fragment="alert(type, message)">
    <div th:class="'alert alert-' + ${type}" th:text="${message}">알림</div>
</div>

<!-- fragment 사용 - 파라미터 전달 -->
<div th:replace="~{layout/base :: alert('success', '저장되었습니다.')}"></div>
<div th:replace="~{layout/base :: alert('error', ${errorMsg})}"></div>
```

---

## 10. 유틸리티 객체

### 문자열 (#strings)

```html
<p th:text="${#strings.toUpperCase(name)}">NAME</p>
<p th:text="${#strings.toLowerCase(name)}">name</p>
<p th:text="${#strings.trim(name)}">trim</p>
<p th:text="${#strings.length(name)}">길이</p>
<p th:if="${#strings.isEmpty(name)}">이름 없음</p>
<p th:if="${#strings.contains(name, '홍')}">홍씨</p>
<p th:text="${#strings.replace(name, '홍', '김')}">변환</p>
<p th:text="${#strings.substring(name, 0, 3)}">앞 3자</p>

<!-- 기본값 처리 -->
<p th:text="${#strings.defaultString(nickname, '닉네임 없음')}">닉네임</p>
```

### 숫자 (#numbers)

```html
<!-- 천 단위 구분 기호 -->
<p th:text="${#numbers.formatInteger(amount, 0, 'COMMA')}">1,000</p>

<!-- 소수점 포맷 -->
<p th:text="${#numbers.formatDecimal(price, 1, 2)}">1,000.00</p>

<!-- 범위 생성 -->
<li th:each="i : ${#numbers.sequence(1, 5)}" th:text="${i}">1</li>
```

### 날짜 (#dates / #temporals)

```html
<!-- java.util.Date -->
<p th:text="${#dates.format(createdAt, 'yyyy-MM-dd HH:mm')}">날짜</p>
<p th:text="${#dates.year(createdAt)}">2025</p>

<!-- java.time.LocalDateTime (Java 8+, #temporals 사용) -->
<p th:text="${#temporals.format(createdAt, 'yyyy-MM-dd HH:mm')}">날짜</p>
<p th:text="${#temporals.day(createdAt)}">일</p>
```

> `#temporals`를 사용하려면 `thymeleaf-extras-java8time` 의존성 추가 또는 Spring Boot 3.x 이상 사용

### 컬렉션 (#lists, #sets, #maps)

```html
<p th:text="${#lists.size(users)}">크기</p>
<p th:if="${#lists.isEmpty(users)}">목록이 없습니다.</p>
<p th:if="${#lists.contains(roles, 'ADMIN')}">관리자 권한 있음</p>
<p th:text="${#lists.sort(items)}">정렬</p>
```

### 객체 (#objects)

```html
<!-- null 체크 -->
<p th:if="${#objects.isNull(user)}">사용자 없음</p>
<p th:if="${#objects.isNotNull(user)}">사용자 있음</p>
```

---

## 11. 메시지 국제화 (i18n)

```properties
# src/main/resources/messages.properties (기본)
welcome.message=환영합니다, {0}님!
user.name.label=이름
error.required={0}은(는) 필수 항목입니다.
```

```properties
# messages_en.properties (영어)
welcome.message=Welcome, {0}!
user.name.label=Name
error.required={0} is required.
```

```yaml
# application.yml
spring:
  messages:
    basename: messages
    encoding: UTF-8
```

```html
<!-- 단순 메시지 -->
<p th:text="#{user.name.label}">이름</p>

<!-- 파라미터 포함 -->
<p th:text="#{welcome.message(${user.name})}">환영합니다</p>

<!-- 파라미터 여러 개 -->
<p th:text="#{error.required(#{user.name.label})}">필수 오류</p>
```

---

## 12. JavaScript 인라인

```html
<script th:inline="javascript">
    // [[${...}]] - 자바스크립트 안전 출력 (자동 이스케이프)
    const userName = [[${user.name}]];
    const userId = [[${user.id}]];

    // 객체를 JSON으로 직렬화
    const userObj = [[${user}]];
    // 결과: const userObj = {"id": 1, "name": "홍길동", "email": "..."};

    // 조건부
    const isAdmin = [[${user.admin}]];

    // th:block으로 서버사이드 로직
    /*[# th:if="${showDebug}"]*/
    console.log("디버그 모드");
    /*[/]*/
</script>
```

---

## 13. Spring Security 통합

```xml
<!-- pom.xml - Thymeleaf Security 통합 의존성 -->
<dependency>
    <groupId>org.thymeleaf.extras</groupId>
    <artifactId>thymeleaf-extras-springsecurity6</artifactId>
</dependency>
```

```html
<html xmlns:th="http://www.thymeleaf.org"
      xmlns:sec="http://www.thymeleaf.org/extras/spring-security">

<!-- 로그인 여부 -->
<div sec:authorize="isAuthenticated()">로그인 상태입니다.</div>
<div sec:authorize="isAnonymous()">로그인이 필요합니다.</div>

<!-- 권한별 표시 -->
<div sec:authorize="hasRole('ADMIN')">관리자 메뉴</div>
<div sec:authorize="hasAnyRole('ADMIN', 'MANAGER')">관리자/매니저 메뉴</div>

<!-- 현재 로그인 사용자 정보 -->
<p>
    안녕하세요,
    <span sec:authentication="name">사용자</span>님!
</p>

<!-- principal 객체의 필드 접근 -->
<p sec:authentication="principal.email">이메일</p>
```

---

## 14. th:block

렌더링 결과에 실제 HTML 태그가 남지 않는 가상 태그입니다.

```html
<!-- 반복과 조건을 한번에 처리할 때 유용 -->
<table>
    <th:block th:each="user : ${users}">
        <tr th:class="${user.admin} ? 'admin'">
            <td th:text="${user.name}">이름</td>
        </tr>
        <tr th:if="${user.hasNote}">
            <td th:text="${user.note}">메모</td>
        </tr>
    </th:block>
</table>

<!-- fragment를 감싸는 래퍼 없이 정의할 때 -->
<th:block th:fragment="scripts">
    <script th:src="@{/js/common.js}"></script>
    <script th:src="@{/js/app.js}"></script>
</th:block>
```

---

## 15. 리다이렉트 & Flash 속성

```java
// Controller - RedirectAttributes로 Flash 메시지 전달
@PostMapping("/users")
public String create(@Valid UserForm form, BindingResult result,
                     RedirectAttributes redirectAttributes) {
    if (result.hasErrors()) {
        return "user/form";
    }
    userService.save(form);
    redirectAttributes.addFlashAttribute("message", "저장되었습니다.");
    redirectAttributes.addFlashAttribute("messageType", "success");
    return "redirect:/users";
}
```

```html
<!-- Flash 메시지 표시 (리다이렉트 후 1회만 표시) -->
<div th:if="${message}"
     th:class="'alert alert-' + ${messageType}"
     th:text="${message}">
</div>
```

---

## 16. 실전 예제 - 사용자 목록/상세/등록

### 목록 (list.html)

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head th:replace="~{layout/base :: head('사용자 목록')}"></head>
<body>
<header th:replace="~{layout/base :: header}"></header>

<main class="container">
    <h1>사용자 목록</h1>

    <!-- Flash 메시지 -->
    <div th:if="${message}" th:class="'alert alert-' + ${messageType}" th:text="${message}"></div>

    <a th:href="@{/users/new}" class="btn">신규 등록</a>

    <!-- 목록 없음 -->
    <p th:if="${#lists.isEmpty(users)}">등록된 사용자가 없습니다.</p>

    <table th:unless="${#lists.isEmpty(users)}">
        <thead>
            <tr>
                <th>번호</th><th>이름</th><th>이메일</th><th>권한</th><th>관리</th>
            </tr>
        </thead>
        <tbody>
            <tr th:each="user, stat : ${users}">
                <td th:text="${stat.count}">1</td>
                <td>
                    <a th:href="@{/users/{id}(id=${user.id})}"
                       th:text="${user.name}">홍길동</a>
                </td>
                <td th:text="${user.email}">email@example.com</td>
                <td th:text="${user.role}">ADMIN</td>
                <td>
                    <a th:href="@{/users/{id}/edit(id=${user.id})}">수정</a>
                    <form th:action="@{/users/{id}(id=${user.id})}" method="post"
                          style="display:inline">
                        <input type="hidden" name="_method" value="DELETE" />
                        <button type="submit"
                                onclick="return confirm('삭제하시겠습니까?')">삭제</button>
                    </form>
                </td>
            </tr>
        </tbody>
    </table>

    <!-- 페이징 -->
    <nav th:if="${page.totalPages > 1}">
        <a th:href="@{/users(page=0)}" th:class="${page.first} ? 'disabled'">처음</a>
        <a th:each="i : ${#numbers.sequence(0, page.totalPages - 1)}"
           th:href="@{/users(page=${i})}"
           th:text="${i + 1}"
           th:class="${i == page.number} ? 'active'">1</a>
        <a th:href="@{/users(page=${page.totalPages - 1})}"
           th:class="${page.last} ? 'disabled'">마지막</a>
    </nav>
</main>

<footer th:replace="~{layout/base :: footer}"></footer>
</body>
</html>
```

### 등록/수정 폼 (form.html)

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head th:replace="~{layout/base :: head('사용자 등록')}"></head>
<body>
<header th:replace="~{layout/base :: header}"></header>

<main class="container">
    <h1 th:text="${userForm.id != null} ? '사용자 수정' : '사용자 등록'">등록</h1>

    <!-- 전체 오류 -->
    <div th:if="${#fields.hasGlobalErrors()}" class="alert alert-error">
        <p th:each="err : ${#fields.globalErrors()}" th:text="${err}">오류</p>
    </div>

    <form th:action="${userForm.id != null} ? @{/users/{id}(id=${userForm.id})} : @{/users}"
          th:object="${userForm}"
          method="post">

        <!-- 수정 시 PUT 메서드 처리 -->
        <input th:if="${userForm.id != null}" type="hidden" name="_method" value="PUT" />

        <div class="form-group">
            <label for="name">이름 *</label>
            <input type="text" th:field="*{name}" class="form-control"
                   th:classappend="${#fields.hasErrors('name')} ? 'is-invalid'" />
            <div class="invalid-feedback" th:errors="*{name}">이름 오류</div>
        </div>

        <div class="form-group">
            <label for="email">이메일 *</label>
            <input type="email" th:field="*{email}" class="form-control"
                   th:classappend="${#fields.hasErrors('email')} ? 'is-invalid'" />
            <div class="invalid-feedback" th:errors="*{email}">이메일 오류</div>
        </div>

        <div class="form-group">
            <label for="role">권한</label>
            <select th:field="*{role}" class="form-control">
                <option th:each="role : ${roles}"
                        th:value="${role}"
                        th:text="${role}">권한</option>
            </select>
        </div>

        <div class="form-check">
            <input type="checkbox" th:field="*{active}" class="form-check-input" />
            <label class="form-check-label" for="active">활성 여부</label>
        </div>

        <div class="mt-3">
            <button type="submit" class="btn btn-primary">저장</button>
            <a th:href="@{/users}" class="btn btn-secondary">취소</a>
        </div>
    </form>
</main>

<footer th:replace="~{layout/base :: footer}"></footer>
</body>
</html>
```
