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

