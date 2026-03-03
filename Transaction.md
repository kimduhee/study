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


특징
+ MiniLM보다 정확
+ 영어 의미 검색 성능 매우 좋음

<pre><code>SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
</code></pre>

| 항목 | 값 |
|:---|:---|
| 차원 | 768 |
| 언어 | 영어 |
| 용도 | 영어 문서 RAG |
