# Apache Kafka

## 1. 개요 (Overview)

Apache Kafka는 LinkedIn에서 개발하고 2011년 오픈소스로 공개된 **분산 이벤트 스트리밍 플랫폼**입니다.
대용량 실시간 데이터를 안정적으로 처리하기 위해 설계되었으며, 현재는 Apache Software Foundation에서 관리합니다.


### 핵심 특징

| 특징 | 설명 |
|------|------|
| **고처리량** | 초당 수백만 건의 메시지 처리 가능 |
| **낮은 지연** | 밀리초 단위의 낮은 응답 지연 |
| **내구성** | 디스크에 메시지를 영속 저장, 재처리 가능 |
| **확장성** | 브로커·파티션 수평 확장으로 처리량 선형 증가 |
| **분산 처리** | 클러스터 구성으로 장애 허용 및 고가용성 보장 |
| **다양한 통합** | 수백 개의 커넥터 생태계 (Kafka Connect) |

### 전통적인 메시지 큐와의 차이

| 구분 | 전통적 MQ (RabbitMQ 등) | Apache Kafka |
|------|------------------------|--------------|
| 메시지 보존 | 소비 후 삭제 | 설정 기간 동안 보존 (재소비 가능) |
| 소비 방식 | Push (브로커 → 소비자) | Pull (소비자가 직접 가져감) |
| 순서 보장 | 큐 단위 | 파티션 단위 |
| 처리량 | 수십만 msg/s | 수백만 msg/s |
| 사용 패턴 | 작업 큐, RPC | 이벤트 스트리밍, 로그 파이프라인 |

---

## 2. 핵심 개념 (Core Concepts)

### 구성 요소

```
Producer ──► Topic (Partition 0) ──► Consumer Group A
              Topic (Partition 1) ──► Consumer Group B
              Topic (Partition 2)
                     │
                  Broker (Kafka Cluster)
                     │
                  ZooKeeper / KRaft (클러스터 메타데이터 관리)
```

| 구성 요소 | 설명 |
|----------|------|
| **Producer** | 메시지를 발행(Publish)하는 클라이언트 |
| **Consumer** | 메시지를 구독(Subscribe)하는 클라이언트 |
| **Consumer Group** | 같은 Topic을 분담 처리하는 Consumer 집합 |
| **Broker** | Kafka 서버 노드, 메시지를 저장·전달 |
| **Cluster** | 여러 Broker로 구성된 Kafka 서버 집합 |
| **Topic** | 메시지를 분류하는 논리적 채널 (RabbitMQ의 Queue 와 유사) |
| **Partition** | Topic을 물리적으로 나눈 단위, 병렬 처리의 기본 단위 |
| **Offset** | Partition 내 메시지 위치를 나타내는 순차 번호 |
| **Replica** | Partition의 복제본, 장애 시 Leader로 승격 |
| **ZooKeeper / KRaft** | 클러스터 메타데이터·리더 선출 관리 |

---

## 3. 동작 원리 (How It Works)

### 3.1 Topic과 Partition

```
Topic: "order-events"

Partition 0: [msg0] [msg3] [msg6] ...   ← offset 0, 1, 2...
Partition 1: [msg1] [msg4] [msg7] ...
Partition 2: [msg2] [msg5] [msg8] ...
```

- **Topic**: 메시지를 분류하는 이름. 하나의 Topic은 여러 Partition으로 구성
- **Partition**: 실제 메시지가 저장되는 순서 로그. 추가만 가능 (append-only)
- **메시지 키**: 동일 키는 항상 같은 Partition에 저장 → 순서 보장
- **키 없음**: Round-robin 방식으로 Partition에 분배

### 3.2 Offset과 메시지 재처리

```
Partition 0:
[offset 0] [offset 1] [offset 2] [offset 3] [offset 4]
                                   ▲
                              Consumer가 여기까지 읽음
                              (commit offset = 3)
```

- Consumer는 자신이 어디까지 읽었는지 **offset**으로 추적
- Kafka는 메시지를 소비해도 삭제하지 않음 (보존 기간 내 재처리 가능)
- **__consumer_offsets** 내부 Topic에 각 Consumer Group의 offset 저장

### 3.3 Consumer Group과 병렬 처리

```
Topic: "orders" (3개 Partition)

Consumer Group A (주문 처리 서비스):
  Consumer A-1 ──► Partition 0
  Consumer A-2 ──► Partition 1
  Consumer A-3 ──► Partition 2

Consumer Group B (분석 서비스):
  Consumer B-1 ──► Partition 0, 1, 2 (모두 처리)
```

- 하나의 Partition은 같은 Consumer Group 내에서 **1개의 Consumer만 담당**
- Consumer 수 > Partition 수이면 초과 Consumer는 유휴(idle) 상태
- **최대 병렬 처리 수 = Partition 수**
- 다른 Consumer Group은 같은 Topic을 독립적으로 소비 (서로 영향 없음)

### 3.4 Replication (복제)

```
Partition 0 (replication factor = 3):
  Broker 1: [Leader]   ← Producer/Consumer가 직접 통신
  Broker 2: [Follower] ← Leader로부터 복제
  Broker 3: [Follower] ← Leader로부터 복제
```

- **Leader**: 해당 Partition의 읽기·쓰기 담당 브로커
- **Follower**: Leader 데이터를 복제, Leader 장애 시 자동 승격
- **ISR (In-Sync Replica)**: Leader와 동기화된 Replica 목록

### 3.5 메시지 전달 보장 수준

| 수준 | 설명 | 중복 가능성 | 손실 가능성 |
|------|------|:---:|:---:|
| **At most once** | 최대 1번 전달 (재시도 없음) | 없음 | 있음 |
| **At least once** | 최소 1번 전달 (재시도로 중복 가능) | 있음 | 없음 |
| **Exactly once** | 정확히 1번 전달 (트랜잭션) | 없음 | 없음 |

### 3.6 ZooKeeper vs KRaft

| 구분 | ZooKeeper 모드 | KRaft 모드 (Kafka 2.8+) |
|------|---------------|------------------------|
| 역할 | 외부 ZooKeeper가 메타데이터 관리 | Kafka 자체 내장 Raft 합의 알고리즘 |
| 운영 복잡도 | ZooKeeper 별도 운영 필요 | Kafka만 운영 (간소화) |
| 권장 버전 | Kafka 3.x 이하 구버전 | Kafka 3.3+ (Production 준비 완료) |

---

## 4. 설치 및 환경 구성

### 4.1 Docker Compose로 로컬 환경 구성

```yaml
# docker-compose.yml
version: "3.8"

services:
  # KRaft 모드 (ZooKeeper 없이 단독 실행)
  kafka:
    image: confluentinc/cp-kafka:7.6.0
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      CLUSTER_ID: "MkU3OEVBNTcwNTJENDM2Qk"

  # 관리 UI
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
    depends_on:
      - kafka
```

```bash
docker-compose up -d
# Kafka UI: http://localhost:8080
```

### 4.2 ZooKeeper 포함 클러스터 구성

```yaml
version: "3.8"

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka-1:
    image: confluentinc/cp-kafka:7.6.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
    depends_on:
      - zookeeper

  kafka-2:
    image: confluentinc/cp-kafka:7.6.0
    ports:
      - "9093:9093"
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
    depends_on:
      - zookeeper

  kafka-3:
    image: confluentinc/cp-kafka:7.6.0
    ports:
      - "9094:9094"
    environment:
      KAFKA_BROKER_ID: 3
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9094
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
    depends_on:
      - zookeeper
```

---

## 5. 개발 방법 (Development)

### 5.1 Spring Boot + Spring Kafka

#### 의존성

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.kafka</groupId>
    <artifactId>spring-kafka</artifactId>
</dependency>
```

#### 설정

```yaml
# application.yml
spring:
  kafka:
    bootstrap-servers: localhost:9092
    producer:
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.springframework.kafka.support.serializer.JsonSerializer
      acks: all                  # 모든 ISR 확인 후 응답 (데이터 손실 방지)
      retries: 3
      properties:
        enable.idempotence: true # 중복 메시지 방지 (exactly once)
    consumer:
      group-id: my-service
      key-deserializer: org.apache.kafka.common.serialization.StringDeserializer
      value-deserializer: org.springframework.kafka.support.serializer.JsonDeserializer
      auto-offset-reset: earliest  # 처음부터 읽기 | latest: 새 메시지만
      enable-auto-commit: false    # 수동 커밋 (메시지 처리 후 커밋)
      properties:
        spring.json.trusted.packages: "*"
```

#### Producer

```java
@Service
@RequiredArgsConstructor
public class OrderEventProducer {

    private final KafkaTemplate<String, OrderEvent> kafkaTemplate;

    // 기본 발행
    public void publish(OrderEvent event) {
        kafkaTemplate.send("order-events", event.getOrderId(), event);
    }

    // 콜백으로 성공/실패 처리
    public void publishWithCallback(OrderEvent event) {
        kafkaTemplate.send("order-events", event.getOrderId(), event)
            .whenComplete((result, ex) -> {
                if (ex != null) {
                    log.error("발행 실패: {}", ex.getMessage());
                } else {
                    log.info("발행 성공 - partition: {}, offset: {}",
                        result.getRecordMetadata().partition(),
                        result.getRecordMetadata().offset());
                }
            });
    }

    // 특정 Partition에 발행
    public void publishToPartition(OrderEvent event, int partition) {
        kafkaTemplate.send("order-events", partition, event.getOrderId(), event);
    }
}
```

#### Consumer

```java
@Service
@RequiredArgsConstructor
public class OrderEventConsumer {

    // 기본 소비
    @KafkaListener(topics = "order-events", groupId = "order-service")
    public void consume(OrderEvent event) {
        log.info("수신: {}", event);
        processOrder(event);
    }

    // 메타데이터 포함 소비
    @KafkaListener(topics = "order-events", groupId = "order-service")
    public void consumeWithMeta(
        @Payload OrderEvent event,
        @Header(KafkaHeaders.RECEIVED_PARTITION) int partition,
        @Header(KafkaHeaders.OFFSET) long offset
    ) {
        log.info("partition={}, offset={}, event={}", partition, offset, event);
    }

    // 수동 Offset 커밋 (enable-auto-commit: false 필요)
    @KafkaListener(topics = "order-events", groupId = "order-service",
                   containerFactory = "manualAckListenerContainerFactory")
    public void consumeManualAck(OrderEvent event, Acknowledgment ack) {
        try {
            processOrder(event);
            ack.acknowledge();  // 처리 성공 후 커밋
        } catch (Exception e) {
            log.error("처리 실패, 재시도 예정: {}", e.getMessage());
            // acknowledge 호출 안 하면 재처리
        }
    }

    // 배치 소비
    @KafkaListener(topics = "order-events", groupId = "order-service-batch",
                   containerFactory = "batchListenerContainerFactory")
    public void consumeBatch(List<OrderEvent> events) {
        log.info("배치 수신: {}건", events.size());
        processBatch(events);
    }
}
```

#### 수동 Offset 커밋 설정

```java
@Configuration
public class KafkaConsumerConfig {

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, OrderEvent>
    manualAckListenerContainerFactory(ConsumerFactory<String, OrderEvent> cf) {
        var factory = new ConcurrentKafkaListenerContainerFactory<String, OrderEvent>();
        factory.setConsumerFactory(cf);
        factory.getContainerProperties().setAckMode(ContainerProperties.AckMode.MANUAL);
        factory.setConcurrency(3);  // 동시 처리 스레드 수 (≤ Partition 수)
        return factory;
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, OrderEvent>
    batchListenerContainerFactory(ConsumerFactory<String, OrderEvent> cf) {
        var factory = new ConcurrentKafkaListenerContainerFactory<String, OrderEvent>();
        factory.setConsumerFactory(cf);
        factory.setBatchListener(true);
        return factory;
    }
}
```

---

### 5.2 Topic 관리

```java
@Configuration
public class KafkaTopicConfig {

    @Bean
    public NewTopic orderEventsTopic() {
        return TopicBuilder.name("order-events")
            .partitions(6)          // 파티션 수 (= 최대 병렬 처리 수)
            .replicas(3)            // 복제본 수 (브로커 수 이하)
            .config(TopicConfig.RETENTION_MS_CONFIG, "604800000")  // 7일 보존
            .config(TopicConfig.CLEANUP_POLICY_CONFIG, "delete")   // delete | compact
            .build();
    }
}
```

---

### 5.3 에러 처리 및 Dead Letter Topic (DLT)

```java
@Configuration
public class KafkaErrorConfig {

    @Bean
    public DefaultErrorHandler errorHandler(KafkaTemplate<String, Object> template) {
        // Dead Letter Topic으로 실패 메시지 이동
        var recoverer = new DeadLetterPublishingRecoverer(template,
            (record, ex) -> new TopicPartition(record.topic() + ".DLT", record.partition()));

        // 재시도 정책: 2초 간격, 최대 3회
        var backoff = new FixedBackOff(2000L, 3);

        return new DefaultErrorHandler(recoverer, backoff);
    }
}
```

```java
// DLT 메시지 처리
@KafkaListener(topics = "order-events.DLT", groupId = "order-dlt-service")
public void handleDlt(
    @Payload OrderEvent event,
    @Header(KafkaHeaders.EXCEPTION_MESSAGE) String errorMsg
) {
    log.error("DLT 메시지 수신 - error: {}, event: {}", errorMsg, event);
    // 알림 발송 또는 DB 저장
}
```

---

### 5.4 트랜잭션 (Exactly Once)

```java
@Configuration
public class KafkaTransactionConfig {

    @Bean
    public KafkaTransactionManager<String, Object> kafkaTransactionManager(
        ProducerFactory<String, Object> pf) {
        return new KafkaTransactionManager<>(pf);
    }
}
```

```java
@Service
@RequiredArgsConstructor
public class TransactionalOrderService {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Transactional("kafkaTransactionManager")
    public void processAndPublish(OrderEvent event) {
        // 여러 Topic 발행이 하나의 트랜잭션으로 처리
        kafkaTemplate.send("order-events", event);
        kafkaTemplate.send("inventory-events", new InventoryEvent(event));
        // 예외 발생 시 두 메시지 모두 롤백
    }
}
```

---

### 5.5 Node.js (kafkajs)

```bash
npm install kafkajs
```

```js
const { Kafka } = require("kafkajs");

const kafka = new Kafka({
  clientId: "my-app",
  brokers: ["localhost:9092"],
});

// Producer
const producer = kafka.producer();
await producer.connect();
await producer.send({
  topic: "order-events",
  messages: [
    { key: "order-1", value: JSON.stringify({ orderId: "order-1", amount: 5000 }) },
  ],
});
await producer.disconnect();

// Consumer
const consumer = kafka.consumer({ groupId: "order-service" });
await consumer.connect();
await consumer.subscribe({ topic: "order-events", fromBeginning: true });

await consumer.run({
  eachMessage: async ({ topic, partition, message }) => {
    console.log({
      partition,
      offset: message.offset,
      value: JSON.parse(message.value.toString()),
    });
  },
});
```

---

### 5.6 CLI 명령어

```bash
# Topic 생성
kafka-topics.sh --bootstrap-server localhost:9092 \
  --create --topic order-events \
  --partitions 6 --replication-factor 1

# Topic 목록 조회
kafka-topics.sh --bootstrap-server localhost:9092 --list

# Topic 상세 정보
kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe --topic order-events

# 메시지 발행 (콘솔)
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic order-events

# 메시지 소비 (처음부터)
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic order-events --from-beginning

# Consumer Group 목록
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list

# Consumer Group lag 확인
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group order-service

# Topic 삭제
kafka-topics.sh --bootstrap-server localhost:9092 \
  --delete --topic order-events
```

---

## 6. Kafka Streams (스트림 처리)

메시지를 소비하면서 실시간으로 변환·집계·필터링하는 라이브러리입니다.

```java
@Configuration
public class OrderStreamConfig {

    @Bean
    public KStream<String, OrderEvent> orderStream(StreamsBuilder builder) {
        KStream<String, OrderEvent> stream = builder.stream("order-events");

        // 필터링: 5만원 이상 주문만
        stream
            .filter((key, event) -> event.getAmount() >= 50000)
            .to("large-order-events");

        // 변환: 이벤트 → 알림 메시지
        stream
            .mapValues(event -> new NotificationMessage(event.getUserId(), "주문 완료"))
            .to("notification-events");

        // 집계: 5분 윈도우로 주문 건수 집계
        stream
            .groupByKey()
            .windowedBy(TimeWindows.ofSizeWithNoGrace(Duration.ofMinutes(5)))
            .count()
            .toStream()
            .to("order-count-events");

        return stream;
    }
}
```

---

## 7. Kafka Connect (데이터 파이프라인)

코드 없이 설정만으로 외부 시스템과 Kafka 간 데이터를 연동합니다.

```
DB / API / S3 ──► Source Connector ──► Kafka ──► Sink Connector ──► DB / ES / S3
```

```json
// Source Connector 등록 (DB → Kafka)
POST /connectors
{
  "name": "mysql-source",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "mysql",
    "database.port": "3306",
    "database.user": "kafka",
    "database.password": "password",
    "database.server.name": "mydb",
    "table.include.list": "mydb.orders",
    "database.history.kafka.topic": "schema-changes.mydb"
  }
}

// Sink Connector 등록 (Kafka → Elasticsearch)
POST /connectors
{
  "name": "elasticsearch-sink",
  "config": {
    "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
    "connection.url": "http://elasticsearch:9200",
    "topics": "order-events",
    "type.name": "_doc",
    "key.ignore": "true"
  }
}
```

---

## 8. 주요 설정 파라미터

### Producer 주요 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `acks` | `1` | `0`: 확인 없음, `1`: Leader만, `all`: 모든 ISR |
| `retries` | `2147483647` | 실패 시 재시도 횟수 |
| `batch.size` | `16384` (16KB) | 배치 전송 크기 |
| `linger.ms` | `0` | 배치 대기 시간 (ms) |
| `compression.type` | `none` | `gzip`, `snappy`, `lz4`, `zstd` |
| `max.in.flight.requests.per.connection` | `5` | 동시 미확인 요청 수 (`1`이면 순서 보장) |
| `enable.idempotence` | `false` | 중복 메시지 방지 (`true` 권장) |

### Consumer 주요 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `group.id` | - | Consumer Group 식별자 (필수) |
| `auto.offset.reset` | `latest` | `earliest`: 처음부터, `latest`: 최신부터 |
| `enable.auto.commit` | `true` | 자동 offset 커밋 여부 |
| `auto.commit.interval.ms` | `5000` | 자동 커밋 주기 |
| `max.poll.records` | `500` | 1회 poll에서 가져올 최대 메시지 수 |
| `session.timeout.ms` | `45000` | Consumer 장애 감지 타임아웃 |
| `fetch.min.bytes` | `1` | 최소 fetch 크기 (높이면 처리량 증가) |

### Topic 주요 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `retention.ms` | `604800000` (7일) | 메시지 보존 기간 |
| `retention.bytes` | `-1` (무제한) | Partition당 최대 보존 크기 |
| `replication.factor` | `1` | 복제본 수 (운영: 3 권장) |
| `min.insync.replicas` | `1` | 최소 동기화 Replica 수 (`acks=all`과 함께 사용) |
| `cleanup.policy` | `delete` | `delete`: 기간 만료 삭제, `compact`: 키 기준 최신만 유지 |

---

## 9. 적용 범위 (Use Cases)

### 9.1 이벤트 기반 MSA

마이크로서비스 간 비동기 이벤트 전달에 활용됩니다.

```
Order Service ──주문생성 이벤트──► Kafka ──► Payment Service
                                         ──► Inventory Service
                                         ──► Notification Service
```

- 서비스 간 느슨한 결합
- SAGA 패턴의 이벤트 버스 역할
- 서비스 장애 시에도 이벤트 유실 없이 복구 후 처리

### 9.2 실시간 로그 파이프라인

```
애플리케이션 로그 ──► Kafka ──► Logstash ──► Elasticsearch ──► Kibana
                          ──► Spark/Flink (실시간 분석)
                          ──► S3 (장기 보관)
```

### 9.3 CDC (Change Data Capture)

DB 변경사항을 실시간으로 캡처하여 다른 시스템에 전파합니다.

```
MySQL Binlog ──► Debezium (Kafka Connect) ──► Kafka ──► Redis (캐시 동기화)
                                                    ──► Elasticsearch (검색 동기화)
                                                    ──► 다른 DB (데이터 복제)
```

### 9.4 실시간 스트림 처리

```
IoT 센서 데이터 ──► Kafka ──► Kafka Streams / Flink ──► 이상 감지 알림
사용자 행동 로그 ──► Kafka ──► 실시간 추천 엔진
결제 데이터     ──► Kafka ──► 사기 탐지 시스템
```

### 9.5 데이터 통합 허브

```
CRM ──►
ERP ──► Kafka (중앙 이벤트 버스) ──► DW (데이터 웨어하우스)
POS ──►                           ──► Data Lake (S3, HDFS)
```

### 9.6 적용 사례 요약

| 도메인 | 활용 사례 |
|--------|----------|
| 이커머스 | 주문·결제·배송 이벤트 처리, 재고 동기화 |
| 금융 | 실시간 사기 탐지, 거래 이벤트 스트리밍 |
| SNS/미디어 | 활동 피드, 좋아요·댓글 알림 파이프라인 |
| IoT | 센서 데이터 수집·분석 |
| 게임 | 사용자 행동 로그, 실시간 랭킹 |
| 모니터링 | 애플리케이션 메트릭·로그 수집 |

---

## 10. 운영 및 모니터링

### 주요 모니터링 지표

| 지표 | 설명 | 임계값 기준 |
|------|------|------------|
| **Consumer Lag** | 미처리 메시지 수 (가장 중요) | 지속 증가 시 경보 |
| **Under-Replicated Partitions** | 복제 지연 Partition 수 | 0이어야 정상 |
| **Active Controller Count** | 클러스터 Controller 수 | 정확히 1이어야 정상 |
| **Request Latency** | Producer/Consumer 요청 지연 | 서비스별 SLA 기준 |
| **Bytes In/Out** | 브로커 네트워크 처리량 | 용량 계획에 활용 |

### Consumer Lag 모니터링

```bash
# Consumer Group Lag 확인
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group order-service

# 출력 예시:
# GROUP         TOPIC         PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# order-service order-events  0          1000            1050            50
# order-service order-events  1          2000            2000            0
```

### Prometheus + Grafana 연동

```yaml
# JMX Exporter로 Kafka 메트릭 수집
# docker-compose.yml에 추가
kafka:
  environment:
    KAFKA_JMX_PORT: 9999
    EXTRA_ARGS: -javaagent:/opt/jmx-exporter/jmx_prometheus_javaagent.jar=7071:/opt/jmx-exporter/kafka.yml
```

---

## 11. Kafka vs 대안 기술 비교

| 구분 | Kafka | RabbitMQ | AWS SQS/SNS | Pulsar |
|------|-------|----------|-------------|--------|
| 처리량 | 매우 높음 | 높음 | 중간 | 매우 높음 |
| 메시지 보존 | 기간 보존 | 소비 후 삭제 | 소비 후 삭제 | 기간 보존 |
| 순서 보장 | 파티션 내 보장 | 큐 단위 보장 | FIFO 큐 선택 | 파티션 내 보장 |
| 운영 복잡도 | 높음 | 중간 | 낮음 (완전관리형) | 높음 |
| 멀티 테넌시 | 제한적 | 제한적 | AWS 계정 단위 | 네이티브 지원 |
| 적합 상황 | 대용량 스트리밍 | 작업 큐·RPC | AWS 환경·서버리스 | 멀티 테넌트·지역 복제 |
