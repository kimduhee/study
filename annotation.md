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


# 설정관리
### @Configuration
설정 클래스

### @Bean
수동 Bean 등록

### @Value
설정값 주입

### @ConfigurationProperties
yml/properties 매핑


# AOP
### @Aspect
해당 클래스가 AOP(공통 기능)을 담당한다고 선언

### @Before
대상 메서드 실행 전에 실행

### @After
메서드 실행 후 무조건 실행(예외 포함)

### @AfterReturning
정상적으로 끝났을 때만 실행

### @AfterThrowing
예외 발생 시 실행

### @Around(중요)
메서드 전체를 감싸서 제어하며 실행 전/후와 직접 실행까지 가능

### @Pointcut
포인트컷을 재사용 가능하게 정의


# 예외처리
###  @ExceptionHandler
특정 예외처리

### @ControllerAdvice
전역 예외처리


 
