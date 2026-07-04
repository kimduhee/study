# MSA (Microservices Architecture)

## 1. 개요 (Overview)

MSA(Microservices Architecture)는 하나의 대형 애플리케이션을 **독립적으로 배포 가능한 소규모 서비스들의 집합**으로 구성하는 소프트웨어 설계 방식입니다.

각 서비스는 특정 비즈니스 기능을 담당하며, 독립적인 프로세스로 실행되고 경량 API(주로 HTTP/REST 또는 메시지 큐)를 통해 통신합니다.



### 모놀리식 vs MSA 비교

| 구분 | 모놀리식 (Monolithic) | MSA (Microservices) |
|------|----------------------|---------------------|
| 배포 단위 | 전체 애플리케이션 | 개별 서비스 |
| 확장 방식 | 전체 스케일 아웃 | 특정 서비스만 스케일 아웃 |
| 기술 스택 | 단일 스택 | 서비스별 독립 선택 |
| 장애 영향 | 전체 서비스 중단 위험 | 특정 서비스만 영향 |
| 개발 속도 | 초기엔 빠름, 규모 커질수록 느림 | 팀별 독립 개발 가능 |
| 복잡도 | 코드베이스 복잡 | 운영/인프라 복잡 |

### MSA 핵심 원칙

- **단일 책임 원칙(Single Responsibility)**: 각 서비스는 하나의 비즈니스 기능에 집중
- **느슨한 결합(Loose Coupling)**: 서비스 간 의존성 최소화
- **높은 응집성(High Cohesion)**: 관련 기능은 동일 서비스 내에 집중
- **독립적 배포(Independent Deployment)**: 각 서비스를 독립적으로 배포 가능
- **분산 데이터 관리(Decentralized Data Management)**: 서비스별 독립 데이터베이스

---

## 2. 아키텍처 구조 (Architecture Structure)

### 전체 구성도

```
┌─────────────────────────────────────────────────────────────────┐
│                          클라이언트                              │
│                  (Web / Mobile / Third-party)                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway                                   │
│         (라우팅 / 인증 / 로드밸런싱 / Rate Limiting)              │
└────┬─────────────┬──────────────┬──────────────┬────────────────┘
     │             │              │              │
     ▼             ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  User   │  │  Order  │  │Product  │  │Payment  │
│ Service │  │ Service │  │ Service │  │ Service │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │             │              │              │
     ▼             ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ User DB │  │Order DB │  │Prod. DB │  │Pay. DB  │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Message Broker      │
              │  (Kafka / RabbitMQ)   │
              └───────────────────────┘
```

### 주요 컴포넌트

#### 2.1 API Gateway
클라이언트 요청의 단일 진입점으로, 모든 서비스 앞단에 위치합니다.

**주요 역할:**
- 라우팅: 요청을 적절한 서비스로 전달
- 인증/인가: JWT 토큰 검증
- 로드 밸런싱: 트래픽 분산
- Rate Limiting: 과도한 요청 제한
- 로깅/모니터링: 요청 추적

#### 2.2 Service Discovery (서비스 디스커버리)
동적으로 서비스 인스턴스를 탐색하고 등록하는 메커니즘입니다.

- **클라이언트 사이드 디스커버리**: 클라이언트가 서비스 레지스트리에서 직접 조회
- **서버 사이드 디스커버리**: 로드 밸런서가 서비스 레지스트리를 조회

#### 2.3 서비스 간 통신 방식

**동기 통신 (Synchronous)**
```
Service A ──HTTP/gRPC──► Service B
         ◄────────────── (응답 대기)
```
- REST API: HTTP 기반, 범용성 높음
- gRPC: Protobuf 기반, 고성능 바이너리 통신

**비동기 통신 (Asynchronous)**
```
Service A ──이벤트 발행──► Message Broker ──이벤트 구독──► Service B
         (응답 대기 없음)
```
- Message Queue: 이벤트 기반 느슨한 결합

#### 2.4 Circuit Breaker (서킷 브레이커)
연속 장애 발생 시 요청을 차단하여 장애 전파를 방지합니다.

```
상태 전환:
  CLOSED ──(실패 임계값 초과)──► OPEN ──(타임아웃 후)──► HALF-OPEN
    ▲                                                        │
    └──────────────────(성공)────────────────────────────────┘
```

- **CLOSED**: 정상 동작, 요청 통과
- **OPEN**: 즉시 실패 반환, 원격 호출 차단
- **HALF-OPEN**: 일부 요청 허용하여 복구 여부 확인

#### 2.5 분산 트랜잭션 패턴

**SAGA 패턴**
서비스 간 데이터 일관성을 보장하는 분산 트랜잭션 방식입니다.
각 단계는 로컬 트랜잭션으로 처리되며, 실패 시 이전 단계를 되돌리는 **보상 트랜잭션(Compensating Transaction)**을 실행합니다.

```
Choreography SAGA (이벤트 기반):
Order Service ──주문생성 이벤트──► Payment Service ──결제성공 이벤트──► Inventory Service
              ◄──결제실패 이벤트── (실패 시 보상 트랜잭션)

Orchestration SAGA (오케스트레이터):
                    ┌────────────────────────────────────┐
                    │          SAGA Orchestrator          │
                    └──┬──────────────┬──────────────┬───┘
                       │              │              │
                       ▼              ▼              ▼
                 Order Service  Payment Service  Inventory Service
```

---

#### 2.6 SAGA 보상 트랜잭션 (Compensating Transaction)

##### 개념

보상 트랜잭션은 **이미 완료된 로컬 트랜잭션을 비즈니스적으로 되돌리는 작업**입니다.
RDBMS의 ROLLBACK과 달리 이미 커밋된 상태를 취소하는 별도의 비즈니스 로직입니다.

> 일반 트랜잭션 T가 있다면, 보상 트랜잭션 C는 T의 효과를 논리적으로 취소합니다.
> 단, 실행 중에 발생한 부수 효과(이메일 발송 등)는 되돌릴 수 없습니다.

##### 정방향 트랜잭션 vs 보상 트랜잭션

| 단계 | 정방향 트랜잭션 | 보상 트랜잭션 |
|------|----------------|--------------|
| 1. 주문 | 주문 생성 (PENDING) | 주문 취소 (CANCELLED) |
| 2. 결제 | 결제 승인, 금액 차감 | 결제 환불, 금액 반환 |
| 3. 재고 | 재고 차감 | 재고 복구 |
| 4. 배송 | 배송 요청 | 배송 취소 요청 |

##### 성공 시나리오 (정방향 흐름)

```
[1] Order Service     주문 생성 (PENDING)
        │
        ▼ 주문생성 이벤트
[2] Payment Service   결제 승인, 금액 차감
        │
        ▼ 결제성공 이벤트
[3] Inventory Service 재고 차감
        │
        ▼ 재고차감 이벤트
[4] Delivery Service  배송 요청
        │
        ▼ 배송요청 이벤트
[5] Order Service     주문 상태 CONFIRMED
```

##### 실패 시나리오 (보상 트랜잭션 흐름)

```
[1] Order Service     주문 생성 (PENDING)          ◄─── [보상] 주문 취소 (CANCELLED)
        │                                                         ▲
        ▼ 주문생성 이벤트                                          │ 주문취소 이벤트
[2] Payment Service   결제 승인, 금액 차감           ◄─── [보상] 결제 환불
        │                                                         ▲
        ▼ 결제성공 이벤트                                          │ 결제환불 이벤트
[3] Inventory Service 재고 차감                      ◄─── [보상] 재고 복구
        │                                                         ▲
        ▼ 재고차감 이벤트                                          │ 재고부족 이벤트
[4] Delivery Service  배송 요청  ── 실패! ──────────────────────────┘
                      (재고 없음)
```

##### Choreography SAGA - 보상 트랜잭션 구현 예시

```java
// 1. Order Service - 주문 생성 및 보상 처리
@Service
public class OrderService {

    // 정방향: 주문 생성
    @Transactional
    public void createOrder(OrderRequest request) {
        Order order = orderRepository.save(Order.of(request, OrderStatus.PENDING));
        eventPublisher.publish(new OrderCreatedEvent(order));
    }

    // 보상: 주문 취소 (결제/재고 실패 이벤트 수신 시 실행)
    @KafkaListener(topics = "payment-failed-events")
    @Transactional
    public void compensateOrder(PaymentFailedEvent event) {
        Order order = orderRepository.findById(event.getOrderId()).orElseThrow();
        order.cancel("결제 실패로 인한 주문 취소");
        orderRepository.save(order);

        // 보상 완료 이벤트 발행
        eventPublisher.publish(new OrderCancelledEvent(order));
    }
}

// 2. Payment Service - 결제 처리 및 보상 처리
@Service
public class PaymentService {

    // 정방향: 결제 승인
    @KafkaListener(topics = "order-created-events")
    @Transactional
    public void processPayment(OrderCreatedEvent event) {
        try {
            Payment payment = paymentGateway.charge(event.getUserId(), event.getAmount());
            paymentRepository.save(payment);
            eventPublisher.publish(new PaymentSucceededEvent(event.getOrderId(), payment));
        } catch (PaymentException e) {
            // 결제 실패 → 보상 트랜잭션 트리거
            eventPublisher.publish(new PaymentFailedEvent(event.getOrderId(), e.getReason()));
        }
    }

    // 보상: 결제 환불 (재고 부족 이벤트 수신 시 실행)
    @KafkaListener(topics = "inventory-insufficient-events")
    @Transactional
    public void refundPayment(InventoryInsufficientEvent event) {
        Payment payment = paymentRepository.findByOrderId(event.getOrderId()).orElseThrow();
        paymentGateway.refund(payment.getPaymentId());
        payment.markRefunded();
        paymentRepository.save(payment);

        // 결제 환불 완료 → 주문 보상 트리거
        eventPublisher.publish(new PaymentFailedEvent(event.getOrderId(), "재고 부족"));
    }
}

// 3. Inventory Service - 재고 처리 및 보상 처리
@Service
public class InventoryService {

    // 정방향: 재고 차감
    @KafkaListener(topics = "payment-succeeded-events")
    @Transactional
    public void decreaseStock(PaymentSucceededEvent event) {
        Inventory inventory = inventoryRepository.findByProductId(event.getProductId())
            .orElseThrow();

        if (inventory.getStock() < event.getQuantity()) {
            // 재고 부족 → 보상 트랜잭션 트리거
            eventPublisher.publish(new InventoryInsufficientEvent(event.getOrderId()));
            return;
        }

        inventory.decrease(event.getQuantity());
        inventoryRepository.save(inventory);
        eventPublisher.publish(new InventoryDecreasedEvent(event.getOrderId()));
    }

    // 보상: 재고 복구 (배송 실패 등 후속 이벤트 수신 시)
    @KafkaListener(topics = "delivery-failed-events")
    @Transactional
    public void restoreStock(DeliveryFailedEvent event) {
        Inventory inventory = inventoryRepository.findByOrderId(event.getOrderId())
            .orElseThrow();
        inventory.restore(event.getQuantity());
        inventoryRepository.save(inventory);

        eventPublisher.publish(new InventoryRestoredEvent(event.getOrderId()));
    }
}
```

##### Orchestration SAGA - 보상 트랜잭션 구현 예시

```java
// SAGA 오케스트레이터 - 전체 흐름 제어
@Service
public class OrderSagaOrchestrator {

    // SAGA 시작
    @Transactional
    public void startOrderSaga(OrderRequest request) {
        SagaState saga = SagaState.builder()
            .sagaId(UUID.randomUUID().toString())
            .orderId(request.getOrderId())
            .status(SagaStatus.STARTED)
            .currentStep(SagaStep.CREATE_ORDER)
            .build();
        sagaRepository.save(saga);

        // 첫 번째 단계 실행
        commandPublisher.send(new CreateOrderCommand(saga.getSagaId(), request));
    }

    // 각 단계 성공 처리
    @KafkaListener(topics = "saga-order-created")
    public void onOrderCreated(OrderCreatedReply reply) {
        SagaState saga = sagaRepository.findBySagaId(reply.getSagaId());
        saga.setCurrentStep(SagaStep.PROCESS_PAYMENT);
        sagaRepository.save(saga);

        commandPublisher.send(new ProcessPaymentCommand(reply.getSagaId(), reply.getAmount()));
    }

    @KafkaListener(topics = "saga-payment-succeeded")
    public void onPaymentSucceeded(PaymentSucceededReply reply) {
        SagaState saga = sagaRepository.findBySagaId(reply.getSagaId());
        saga.setCurrentStep(SagaStep.DECREASE_INVENTORY);
        sagaRepository.save(saga);

        commandPublisher.send(new DecreaseInventoryCommand(reply.getSagaId(), reply.getProductId()));
    }

    // 실패 시 보상 트랜잭션 체인 실행
    @KafkaListener(topics = "saga-inventory-failed")
    public void onInventoryFailed(InventoryFailedReply reply) {
        SagaState saga = sagaRepository.findBySagaId(reply.getSagaId());
        saga.setStatus(SagaStatus.COMPENSATING);
        saga.setCurrentStep(SagaStep.REFUND_PAYMENT); // 역순으로 보상 시작
        sagaRepository.save(saga);

        // 결제 환불 보상 명령 전송
        commandPublisher.send(new RefundPaymentCommand(reply.getSagaId()));
    }

    @KafkaListener(topics = "saga-payment-refunded")
    public void onPaymentRefunded(PaymentRefundedReply reply) {
        SagaState saga = sagaRepository.findBySagaId(reply.getSagaId());
        saga.setCurrentStep(SagaStep.CANCEL_ORDER); // 다음 보상 단계
        sagaRepository.save(saga);

        // 주문 취소 보상 명령 전송
        commandPublisher.send(new CancelOrderCommand(reply.getSagaId()));
    }

    @KafkaListener(topics = "saga-order-cancelled")
    public void onOrderCancelled(OrderCancelledReply reply) {
        SagaState saga = sagaRepository.findBySagaId(reply.getSagaId());
        saga.setStatus(SagaStatus.COMPENSATED); // 보상 완료
        sagaRepository.save(saga);
    }
}
```

##### SAGA 상태 관리

```
SAGA 상태 전환:

STARTED
  │
  ├──(모든 단계 성공)──► COMPLETED
  │
  └──(특정 단계 실패)──► COMPENSATING
                              │
                              ├──(모든 보상 성공)──► COMPENSATED
                              │
                              └──(보상도 실패)──► COMPENSATION_FAILED
                                                  (수동 처리 필요)
```

```java
public enum SagaStatus {
    STARTED,              // SAGA 시작
    COMPLETED,            // 모든 단계 성공 완료
    COMPENSATING,         // 보상 트랜잭션 진행 중
    COMPENSATED,          // 보상 완료 (롤백 성공)
    COMPENSATION_FAILED   // 보상 실패 (수동 개입 필요)
}

public enum SagaStep {
    // 정방향 단계
    CREATE_ORDER,
    PROCESS_PAYMENT,
    DECREASE_INVENTORY,
    REQUEST_DELIVERY,

    // 보상 단계 (역순)
    CANCEL_DELIVERY,
    RESTORE_INVENTORY,
    REFUND_PAYMENT,
    CANCEL_ORDER
}
```

##### 보상 트랜잭션 설계 원칙

| 원칙 | 설명 |
|------|------|
| **멱등성(Idempotency)** | 동일 보상 트랜잭션이 여러 번 실행되어도 결과가 같아야 함 |
| **역순 실행** | 정방향 실행의 역순으로 보상 트랜잭션을 실행 |
| **완료 보장** | 보상 트랜잭션은 반드시 성공해야 함 (재시도 + 데드레터 큐) |
| **부수효과 고려** | 이메일, SMS 등 외부 발송은 취소 불가 → 별도 안내 메시지 발송 |
| **상태 추적** | SAGA 상태를 DB에 영속화하여 장애 후 재개 가능하게 관리 |

```java
// 멱등성 보장 예시 - 중복 보상 실행 방지
@Transactional
public void refundPayment(String orderId) {
    Payment payment = paymentRepository.findByOrderId(orderId).orElseThrow();

    // 이미 환불된 경우 중복 처리 방지
    if (payment.getStatus() == PaymentStatus.REFUNDED) {
        return;
    }

    paymentGateway.refund(payment.getPaymentId());
    payment.markRefunded();
    paymentRepository.save(payment);
}
```

##### Choreography vs Orchestration 비교

| 구분 | Choreography (이벤트 기반) | Orchestration (중앙 제어) |
|------|--------------------------|--------------------------|
| 제어 방식 | 각 서비스가 이벤트에 반응 | 오케스트레이터가 명령 전달 |
| 결합도 | 느슨한 결합 | 오케스트레이터에 집중 |
| 가시성 | 흐름 파악 어려움 | 흐름 한눈에 파악 가능 |
| 보상 처리 | 각 서비스가 보상 이벤트 처리 | 오케스트레이터가 역순 명령 |
| 복잡도 | 서비스 수 증가 시 복잡 | 오케스트레이터 단일 복잡점 |
| 적합 상황 | 단순한 흐름, 소수 서비스 | 복잡한 흐름, 다수 서비스 |

---

## 3. 주요 오픈소스 생태계 (Open Source Ecosystem)

### 3.1 서비스 프레임워크

#### Spring Boot / Spring Cloud (Java)
마이크로서비스 개발을 위한 가장 널리 사용되는 Java 프레임워크입니다.

```xml
<!-- Spring Cloud 주요 의존성 -->
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-circuitbreaker-resilience4j</artifactId>
</dependency>
```

| 컴포넌트 | 설명 |
|----------|------|
| Spring Boot | 독립 실행형 마이크로서비스 개발 기반 |
| Spring Cloud Gateway | API Gateway 구현 |
| Spring Cloud Eureka | 서비스 디스커버리/레지스트리 |
| Spring Cloud Config | 중앙화된 설정 관리 |
| Spring Cloud Sleuth | 분산 트레이싱 (Zipkin 연동) |
| Resilience4j | Circuit Breaker, Rate Limiter |
| Spring Cloud OpenFeign | 선언형 REST 클라이언트 |

#### Quarkus (Java/Kotlin)
GraalVM 네이티브 이미지를 지원하는 경량 Java 프레임워크입니다.
- 컨테이너 환경에 최적화
- 빠른 시작 시간, 낮은 메모리 사용량

#### Micronaut (Java/Kotlin/Groovy)
컴파일 타임 DI를 사용하는 경량 마이크로서비스 프레임워크입니다.

---

### 3.2 API Gateway

#### Kong Gateway
Nginx 기반의 고성능 API Gateway입니다.

```yaml
# Kong 라우팅 설정 예시
services:
  - name: user-service
    url: http://user-service:8080
    routes:
      - name: user-route
        paths:
          - /api/users
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 100
```

| 기능 | 설명 |
|------|------|
| 라우팅 | URL 패턴 기반 서비스 라우팅 |
| 인증 | JWT, OAuth2, API Key 지원 |
| Rate Limiting | 요청 수 제한 |
| 로깅 | 요청/응답 로깅 |
| 플러그인 생태계 | 다양한 오픈소스 플러그인 지원 |

#### Netflix Zuul / Spring Cloud Gateway
Spring 생태계에 통합된 API Gateway입니다.

```java
// Spring Cloud Gateway 라우팅 설정
@Bean
public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
    return builder.routes()
        .route("user-service", r -> r
            .path("/api/users/**")
            .filters(f -> f.addRequestHeader("X-Request-Source", "gateway"))
            .uri("lb://USER-SERVICE"))
        .route("order-service", r -> r
            .path("/api/orders/**")
            .uri("lb://ORDER-SERVICE"))
        .build();
}
```

---

### 3.3 서비스 디스커버리 & 설정 관리

#### Netflix Eureka
서비스 등록 및 탐색을 위한 REST 기반 서비스입니다.

```yaml
# Eureka 서버 설정
eureka:
  instance:
    hostname: localhost
  client:
    register-with-eureka: false
    fetch-registry: false
    service-url:
      defaultZone: http://${eureka.instance.hostname}:8761/eureka/
```

#### HashiCorp Consul
서비스 디스커버리, 헬스체크, KV 스토어를 제공합니다.

```hcl
# Consul 서비스 등록
service {
  name = "user-service"
  id   = "user-service-1"
  port = 8080
  tags = ["v1", "primary"]
  
  check {
    http     = "http://localhost:8080/health"
    interval = "10s"
    timeout  = "2s"
  }
}
```

---

### 3.4 메시지 브로커 (Message Broker)

#### Apache Kafka
대용량 실시간 스트리밍을 위한 분산 메시지 플랫폼입니다.

```java
// Kafka Producer 예시
@Service
public class OrderEventProducer {
    @Autowired
    private KafkaTemplate<String, OrderEvent> kafkaTemplate;

    public void publishOrderCreated(Order order) {
        OrderEvent event = new OrderEvent("ORDER_CREATED", order);
        kafkaTemplate.send("order-events", order.getId(), event);
    }
}

// Kafka Consumer 예시
@KafkaListener(topics = "order-events", groupId = "payment-service")
public void handleOrderEvent(OrderEvent event) {
    if ("ORDER_CREATED".equals(event.getType())) {
        paymentService.processPayment(event.getOrder());
    }
}
```

#### RabbitMQ
AMQP 기반의 메시지 브로커로, Exchange/Queue/Binding 구조를 사용합니다.

```
Producer ──► Exchange ──(Routing Key)──► Queue ──► Consumer
              │
              ├──► Direct Exchange   (정확한 키 매칭)
              ├──► Topic Exchange    (패턴 매칭)
              ├──► Fanout Exchange   (브로드캐스트)
              └──► Headers Exchange  (헤더 기반)
```

| 비교 항목 | Kafka | RabbitMQ |
|----------|-------|----------|
| 처리량 | 매우 높음 (백만 msg/s) | 높음 (수십만 msg/s) |
| 메시지 보존 | 기간 기반 보존 | 소비 후 삭제 |
| 순서 보장 | 파티션 내 보장 | 큐 내 보장 |
| 사용 사례 | 이벤트 스트리밍, 로그 | 작업 큐, RPC |

---

### 3.5 컨테이너 & 오케스트레이션

#### Docker
애플리케이션을 컨테이너로 패키징합니다.

```dockerfile
# 멀티 스테이지 빌드 예시
FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn clean package -DskipTests

FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
```

#### Kubernetes (K8s)
컨테이너 오케스트레이션의 사실상 표준입니다.

```yaml
# 마이크로서비스 Deployment 예시
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
    spec:
      containers:
        - name: user-service
          image: user-service:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /actuator/health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: user-service
spec:
  selector:
    app: user-service
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
```

| K8s 핵심 리소스 | 설명 |
|----------------|------|
| Pod | 컨테이너 실행 단위 |
| Deployment | Pod 배포 및 롤링 업데이트 관리 |
| Service | Pod에 대한 안정적인 네트워크 엔드포인트 |
| Ingress | 외부 트래픽 라우팅 |
| ConfigMap | 설정 데이터 관리 |
| Secret | 민감 정보 관리 |
| HPA | 수평적 자동 스케일링 |

---

### 3.6 서비스 메시 (Service Mesh)

#### Istio
서비스 간 통신을 관리하는 인프라 레이어입니다.

```yaml
# Istio Virtual Service - 트래픽 분할 (카나리 배포)
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
    - user-service
  http:
    - match:
        - headers:
            cookie:
              regex: "^(.*?;)?(user=test)(;.*)?$"
      route:
        - destination:
            host: user-service
            subset: v2
    - route:
        - destination:
            host: user-service
            subset: v1
          weight: 90
        - destination:
            host: user-service
            subset: v2
          weight: 10
```

**Istio 주요 기능:**
- mTLS 기반 서비스 간 암호화 통신
- 트래픽 관리 (카나리, 블루/그린 배포)
- 관찰 가능성 (메트릭, 로그, 트레이싱)
- Circuit Breaker, Retry, Timeout

#### Linkerd
경량 서비스 메시로 Rust 기반 데이터 플레인을 사용합니다.

---

### 3.7 관찰 가능성 (Observability)

MSA에서는 **로그, 메트릭, 트레이싱** 세 가지 관찰 가능성이 핵심입니다.

#### 분산 트레이싱

**Jaeger / Zipkin**
서비스 간 요청 흐름을 추적합니다.

```
Request: GET /api/orders/123
│
├─► API Gateway (10ms)
│     │
│     ├─► Order Service (25ms)
│     │     │
│     │     ├─► User Service (8ms) ──► User DB
│     │     │
│     │     └─► Product Service (12ms) ──► Product DB
│     │
│     └─► Payment Service (15ms) ──► Payment DB
│
Total: 70ms
```

#### 메트릭 수집

**Prometheus + Grafana**

```yaml
# Prometheus 스크레이핑 설정
scrape_configs:
  - job_name: 'spring-actuator'
    metrics_path: '/actuator/prometheus'
    static_configs:
      - targets:
          - 'user-service:8080'
          - 'order-service:8080'
          - 'product-service:8080'
```

**주요 모니터링 지표:**
- RED 메트릭: Rate(요청 수), Error(에러율), Duration(응답 시간)
- USE 메트릭: Utilization(활용률), Saturation(포화도), Errors(에러)

#### 로그 집계

**ELK Stack (Elasticsearch + Logstash + Kibana)**

```
각 서비스 로그 ──► Filebeat/Logstash ──► Elasticsearch ──► Kibana 대시보드
```

**EFK Stack (Elasticsearch + Fluentd + Kibana)**

```
각 서비스 로그 ──► Fluentd ──► Elasticsearch ──► Kibana 대시보드
```

---

### 3.8 CI/CD 파이프라인

#### GitOps with ArgoCD

```
개발자 코드 Push
    │
    ▼
GitHub Actions/Jenkins (빌드 & 테스트)
    │
    ▼
Docker Image Build & Push (Docker Hub/ECR)
    │
    ▼
GitOps 저장소 업데이트 (K8s 매니페스트)
    │
    ▼
ArgoCD (변경 감지 & 자동 배포)
    │
    ▼
Kubernetes 클러스터 배포
```

---

## 4. MSA 설계 패턴 (Design Patterns)

### 4.1 데이터 관리 패턴

| 패턴 | 설명 | 사용 사례 |
|------|------|----------|
| Database per Service | 서비스별 독립 DB | 서비스 독립성 보장 |
| Shared Database | 여러 서비스가 DB 공유 | 점진적 마이그레이션 |
| CQRS | 읽기/쓰기 모델 분리 | 복잡한 쿼리, 고성능 |
| Event Sourcing | 이벤트로 상태 저장 | 감사 로그, 이벤트 재생 |

### 4.2 통신 패턴

| 패턴 | 설명 |
|------|------|
| API Gateway | 단일 진입점, 요청 라우팅 |
| BFF (Backend for Frontend) | 클라이언트별 맞춤형 게이트웨이 |
| Service Mesh | 인프라 레벨 통신 관리 |
| Saga | 분산 트랜잭션 관리 |

### 4.3 안정성 패턴

| 패턴 | 설명 |
|------|------|
| Circuit Breaker | 연쇄 장애 방지 |
| Retry with Backoff | 일시적 오류 재시도 |
| Bulkhead | 장애 격리 (스레드 풀 분리) |
| Timeout | 응답 지연 제한 |
| Health Check | 서비스 상태 모니터링 |

---

## 5. MSA 도입 시 고려사항

### 장점
- 서비스별 독립 배포 및 스케일링 가능
- 기술 스택 자유로운 선택
- 팀별 자율적 개발 및 빠른 출시
- 장애 격리로 시스템 안정성 향상

### 단점 및 과제
- 분산 시스템 운영 복잡도 증가
- 서비스 간 통신 오버헤드
- 분산 트랜잭션 처리 어려움
- 테스트 및 디버깅 복잡도 증가
- 초기 인프라 구축 비용

### MSA 도입 권장 상황
- 팀 규모가 크고 독립적 개발 조직이 필요한 경우
- 서비스별 다른 스케일링 요구사항이 있는 경우
- 빠른 배포 주기가 필요한 경우
- 다양한 기술 스택 사용이 필요한 경우

### MSA 비권장 상황
- 소규모 팀, 단순한 도메인
- 초기 스타트업 (복잡도보다 빠른 개발 우선)
- MSA 운영 전문성이 부족한 경우

---

## 6. 오픈소스 도구 전체 요약

| 카테고리 | 도구 | 설명 |
|----------|------|------|
| **프레임워크** | Spring Boot/Cloud | Java MSA 개발 표준 |
| | Quarkus | 경량 Java, 네이티브 이미지 |
| | Micronaut | 컴파일 타임 DI |
| **API Gateway** | Kong | Nginx 기반 고성능 게이트웨이 |
| | Spring Cloud Gateway | Spring 생태계 게이트웨이 |
| | AWS API Gateway | 완전관리형 클라우드 게이트웨이 |
| **서비스 디스커버리** | Eureka | Netflix OSS, Spring 통합 |
| | Consul | HashiCorp, 헬스체크 포함 |
| | etcd | K8s 내장 KV 스토어 |
| **메시지 브로커** | Apache Kafka | 고처리량 스트리밍 |
| | RabbitMQ | AMQP 메시지 큐 |
| | Apache Pulsar | 멀티 테넌트 메시징 |
| **컨테이너** | Docker | 컨테이너 표준 |
| | Kubernetes | 컨테이너 오케스트레이션 |
| | Helm | K8s 패키지 매니저 |
| **서비스 메시** | Istio | 기능 풍부한 서비스 메시 |
| | Linkerd | 경량 서비스 메시 |
| **모니터링** | Prometheus | 메트릭 수집 |
| | Grafana | 메트릭 시각화 |
| | Jaeger | 분산 트레이싱 |
| | Zipkin | 분산 트레이싱 |
| **로그** | ELK Stack | Elasticsearch+Logstash+Kibana |
| | Fluentd | 로그 수집기 |
| **CI/CD** | Jenkins | 파이프라인 자동화 |
| | ArgoCD | GitOps 기반 배포 |
| | Tekton | 클라우드 네이티브 CI/CD |
| **Circuit Breaker** | Resilience4j | Java Circuit Breaker |
| | Hystrix | Netflix OSS (deprecated) |
| **설정 관리** | Spring Cloud Config | Spring 설정 서버 |
| | HashiCorp Vault | 시크릿 관리 |
