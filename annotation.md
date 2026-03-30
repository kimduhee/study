# 기본설정/시작
### @SpringBootApplication
아래 3개를 포함하는 어너테이션
+ @Configuration
+ @EnableAutoConfiguration
+ @ComponentScan
<pre><code>@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
</code></pre>

# Bean 등록/의존성 주입
### @Component
일반적인 Bean 등록
<pre><code>@Component
public class LoginService {
}
</code></pre>

### @Service
서비스 로직 등록
<pre><code>@Service
public class PayService {
}
</code></pre>

### @Repository
DB접근 계층

### @Controller
view 반환
<pre><code>@Controller
public class PayController {
}
</code></pre>

### @RestController
json 반환
<pre><code>@RestController
public class PayController {
}
</code></pre>

### @Autowired
Bean 자동주입(필드 주입 방식)
=> 최근에는 생성자 주입방식 권장
<pre><code>@RestController
public class PayController {
    @Autowired
    private PayService payService;
}
</code></pre>

### @Qualifier
같은 타입의 Bean이 여러개 있을때, 어떤 Bean을 사용할지 명확히 지정하는 어노테이션
<pre><code>@Qualifier(payService)
private PayService payService {
}
</code></pre>

### @Primary
동일 타입에서 기본값으로 지정
<pre><code>@Primary
public class PayServiceImpl implements PayService {
}
</code></pre>


# 웹 요청 처리
### @RequestMapping
기본 URL 매핑
<pre><code>@RestController
@RequestMapping("/api")
public class PayController {
    @RequestMapping(value="/test", method=RequestMethod.POST)
    public String test() {
        return "test";
    }
}
</code></pre>

### @GetMapping, @PostMapping, @PutMapping, @DeleteMapping
http별 메서드 맵핑
<pre><code>@PostMapping("/test")
public String test() {
    return "test";
}
</code></pre>

### @RequestParam
쿼리 파라미터
<pre><code>@PostMapping("/test")
public String test(@RequestParam String id) {
    return "test";
}
</code></pre>

### @PathVariable
url 경로값을 변수로 사용
<pre><code>@GetMapping("/test/{id}")
public String test(@PathVariable Long id) {
    return "test";
}
</code></pre>

### @RequestBody
body에 담긴 json 객체반환
<pre><code>@PostMapping("/test")
public String test(@RequestBody Test test) {
    return "test";
}
</code></pre>


# DB/JPA
### @Entity
DB 테이블 맵핑

### @Id
PK 설정

### @GeneratedValue
PK 자동생성

### @Transactional
트랜잭션 처리
<pre><code>@Service
public class PayService {
    @Transactional
    public void myPay() {
    }
}

롤백
@Transactional(rollbackFor = Exception.class)

읽기 전용
@Transactional(readOnly = true)

전파 옵션(Propergation)
@Transactional(propergation = Propagation.REQUIRED)

격리 수준(Isolation)
@Transactional(isolation = Isolation.READ_COMMITTED)
</code></pre>

+ 전파 옵션(Propagation)

| 옵션 | 설명 |
|:---|:---|
| REQUIRED | 기본값(기준에 있으면 참여) |
| REQUIRES_NEW | 항상 새 트랜잭션 |
| SUPPORTS | 있으면 참여, 없으면 없음 |

+ 격리 수준(Isolation)

| 수준 | 설명 |
|:---|:---|
| READ_UNCOMMITTED | Dirty Read 가능 |
| READ_COMMITTED | 커밋된 데이터만 |
| REPEATABLE_READ | 반복 조회 동일 |
| SERIALIZABLE | 완전 격리 |

### @Mapper
이 인터페이스가 Mybatis Mapper라는걸 정의
<pre><code>@Mapper
public interface LoginMapper {
}
</code></pre>


# 설정관리
### @Configuration
설정 클래스
<pre><code>@Configuration
public class AppConfig {
}
</code></pre>

### @Bean
수동 Bean 등록
<pre><code>@Configuration
public class AppConfig {
    @Bean
    public LoginService loginService() {
        return new LoginServiceImpl();
    }
}
</code></pre>

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


 
