# 기본개념
트랜잭션은 ACID 특성을 가진다.
+ Atomicity(원자성): 모두 성공하거나 모두 실패
+ Consistency(일관성): 트랜잭션 전후 전후 데이터 일관성 유지
+ Isolation(격리성): 동시에 실행되는 트랜잭션 간 간섭 방지
+ Durability(지속성): commit 이후 데이터는 영구 저장


# Spring에서 트랜잭션 사용방법
Spring에서는 보통 선언적 트랜잭션을 사용한다.
<pre><code>@Transactional 사용
@Service
public class OrderService {
  @Transactional
  public void order() {
    //재고 감소
    //주문 생성
    //결제 처리
  }
}
</code></pre>
+ 메서드가 실행되기 전에 트랜잭션 시작
+ 정상 종료: commit
+ RuntimeException 발생: rollback


# 트랜잭션 동작 방식
Spring은 내부적으로 AOP(프록시)를 사용
+ 프록시 객체 생성
+ 메서드 호출 시 
> 트랜잭션 시작<br>
> 실제 메서드 실행<br>
> 예외 여부 확인<br>
> commit 또는 rollback
<pre><code>
public void methodA() {
  methodB(); //트랜잭션 적용 안됨
}

@Transactional
public void methodB() {}
</code></pre>


# rollback 기준
기본 동작
+ RuntimeException => rollback
+ Error => rollback
+ Checked Exception => commit
Checked Exception도 rollback 필요시
<pre><code>
@Transactional(rollbackFor = Exception.class)
</code></pre>


#주요옵션

### 1)propagation(전파 옵션)
| 옵션 | 설명 |
|:---|:---|
| REQUIRED(기본) | 기존 트랜잭션 있으면 참여, 없으면 생성 |
| REQUIRES_NEW | 항상 새 트랜잭션 생성 |
| SUPPORTS | 있으면 참여, 없으면 그냥 실행 |
| MANDATORY | 반드시 기존 트랜잭션 필요 |
| NEVER | 트랜잭션 없이 실행 |
| NESTED | 중첩 트랜잭션 |
<pre><code>
@Transactional(propagation = Propagation.REQUIRES_NEW)
</code></pre>

### 2)isolation(격리 수준)
DB 격리 수준 설정
| 수준 | 설명 |
|:---|:---|
| READ_UNCOMMITTED | Dirty Read 가능 |
| READ_COMMITTED | 커밋된 데이터만 읽음 |
| REPEATABLE_READ | 반복 조회 시 동일 값 |
| SERIALIZABLE | 완전 격리(성능 낮음) |
<pre><code>
@Transactional(isolation = Isolation.REPEATABLE_READ)
</code></pre>

### 3)readOnly
조회 전용 최적화
<pre><code>
@Transactional(readOnly = true)
</code></pre>
+ Hibernate flush 방지
+ 성능 향상
