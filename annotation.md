# 기본설정/시작
### @SpringBootApplication
아래 3개를 포함하는 어너테이션
+ @Configuration
+ @EnableAutoConfiguration
+ @ComponentScan

# Bean 등록/의존성 주입
### @Component
일반적인 Bean 등록

### @Service
서비스 로직 등록

### @Repository
DB접근 계층

### @Controller
view 반환

### @RestController
json 반환

### @Autowired
Bean 자동주입(필드 주입 방식)
=> 최근에는 생성자 주입방식 권장


# 웹 요청 처리
### @RequestMapping
기본 URL 매핑

### @GetMapping, @PostMapping, @PutMapping, @DeleteMapping
http별 메서드 맵핑

### @RequestParam
쿼리 파라미터

### @PathVariable
url 경로값

### @RequestBody
body에 담긴 json 객체반환


# DB/JPA
### @Entity
DB 테이블 맵핑

### @Id
PK 설정

### @GeneratedValue
PK 자동생성

### @Transactional
트랜잭션 처리

### @Mapper
이 인터페이스가 Mybatis Mapper라는걸 정의
<pre><code>@Mapper
public interface LoginMapper {
}
</code></pre>


# 설정관리
### @Configuration
설정 클래스

### @Bean
수동 Bean 등록

### @Value
설정값 주입
<pre><code>@Value("${login.url}")
private String loginUrl;
</code></pre>

### @ConfigurationProperties
yml/properties 매핑


# AOP
### @Aspect
해당 클래스가 AOP(공통 기능)을 담당한다고 선언
<pre><code>@Aspect
@Component
public class LoggingAspect {
}
</code></pre>

### @Before
대상 메서드 실행 전에 실행
<pre><code>@Before("execution(* com.sample.service.*.*(..))")
public void beforeLog() {
    //메서드 실행 전
}
</code></pre>
+ \* => 모든 리턴 타입
+ com.sample.service => 해당 패키지
+ \*.\* => 모든 클래스 + 모든 메서드
+ (..) => 모든 파라미터

### @After
메서드 실행 후 무조건 실행(예외 포함)
<pre><code>@After("execution(* com.sample.service.*.*(..))")
public void afterLog() {
    //메서드 실행 
}
</code></pre>

### @AfterReturning
정상적으로 끝났을 때만 실행
<pre><code>@AfterReturning(pointcut = "execution(* com.sample.service.*.*(..))", returning = "result")
public void afterReturningLog(Object result) {
    System.out.println("리턴값:" + result);
}
</code></pre>

### @AfterThrowing
예외 발생 시 실행
<pre><code>@AfterThrowing(pointcut = "execution(* com.sample.service.*.*(..))", throwing = "ex")
public void exceptionLog(Exception ex) {
    System.out.println("예외:" + ex.getMessage());
}
</code></pre>

### @Around(중요)
메서드 전체를 감싸서 제어하며 실행 전/후와 직접 실행까지 가능
<pre><code>@Around("execution(* com.sample.service.*.*(..))")
public void Object aroundLog(ProceedingJoinPoing joinPoint) throws Throwable {
    System.out.println("실행전...");
    Object result = joinPoint.proceed();
    System.out.println("실행후...");
    return result;
}
</code></pre>

### @Pointcut
포인트컷을 재사용 가능하게 정의
<pre><code>@APointcut("execution(* com.sample.service.*.*(..))")
public void serviceMethods() {}
</code></pre>

<pre><code>사용
@Before("serviceMethods()")
</code></pre>


# 필터 관련
### @WebFilter
서블릿 필터 등록
<pre><code>@WebFilter(urlPatterns = "/*")
public class LoggingFilter implements Filter {
}
</code></pre>

### @ServletComponentScan
@WebFilter, @WebServlet, @WebListener 활성화
<pre><code>@SpringBootApplication
@ServletComponentScan
public class Application {
}
</code></pre>

### OncePerRequestFilter
요청당 한 번만 실행 보장
<pre><code>public class JwtFilter extends OncePerRequestFilter {
    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) {
        filterChain.doFilter(request, response);
    }
}
</code></pre>


# 예외처리
###  @ExceptionHandler
특정 예외처리

### @ControllerAdvice
전역 예외처리


 
