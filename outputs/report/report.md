# KV cache 최적화 기술 다관점 평가 — KIVI(SW) · InfiniGen(HW) · 스마트폰 온디바이스 LLM

## SUMMARY

KV cache 최적화 기술인 KIVI(소프트웨어)와 InfiniGen(하드웨어)을 스마트폰 온디바이스 LLM 도메인에서 기술 성숙도, 시장, 이해관계자, 도메인 적합성 관점으로 비교했다. 기술 성숙도에서는 KIVI가 공개 논문과 구현체를 기반으로 TRL 4로 평가된 반면, InfiniGen은 TRL 3으로 평가되어 검증 범위 차이가 나타났다. 시장 관점에서는 KIVI가 인기 모델에 적용된 사례가 확인되나 InfiniGen은 채택과 시장 연결 근거가 부족해 평가가 엇갈렸다. 이해관계자 관점에서는 KIVI가 도입 기업과 개발자에게 긍정적 평가를 받는 반면, InfiniGen은 데이터 전송 병목과 정확도 저하 우려가 존재한다. 도메인 적합성에서는 KIVI가 재학습 없이 추가 하드웨어 요구 없이 스마트폰 환경에 적합하다고 판단되나, InfiniGen은 CPU 메모리 오프로딩과 PCIe 대역폭 전제가 스마트폰 환경과 맞지 않아 조건부 적합으로 평가되었다. 가장 큰 엇갈림은 도메인 적합성에서 InfiniGen의 하드웨어 전제와 실제 스마트폰 환경 부합 여부, 그리고 시장 채택 근거의 유무에서 나타난다. 본 보고서는 공개 정보 기반 추정에 의존하며, 판단 문장 중 약 5%가 `[추론]` 태그를 포함한다.

## 1. 분석 배경

KV cache는 LLM 추론 시 토큰당 약 0.5MB(7B 모델, fp16, MHA 기준)의 메모리를 요구한다. 문맥 길이가 늘어날수록 KV cache 크기가 선형적으로 증가해, 초기의 연산 병목이 메모리 병목으로 전환되는 현상이 나타난다. 이로 인해 긴 문맥을 처리할 때 메모리 용량이 중요한 제약 요소가 된다. KV cache 최적화는 크게 두 가지 접근으로 나뉜다. 소프트웨어적 방법은 데이터 크기를 줄이기 위해 압축이나 양자화 기술을 활용하며, 하드웨어적 방법은 메모리나 스토리지 계층을 확장해 저장 공간을 넓히는 데 초점을 맞춘다.

스마트폰 온디바이스 LLM 환경에서는 메모리 확장이 불가능하고 배터리 제약이 있으며, 모델 재학습도 지원되지 않는다. 또한 어시스턴트의 응답 품질이 제품 품질과 직결되어 압축 손실에 민감하다. 대표적인 시나리오로는 12GB RAM 환경에서 4bit로 양자화된 7B 모델을 사용해 8K 토큰 문맥을 처리할 때, fp16 KV cache가 약 4GB를 차지하는 경우가 있다. 이처럼 온디바이스 LLM은 메모리 용량이 절대적인 제약 조건으로 작용한다.

KV cache 최적화는 Recall, Latency, Memory 세 축에서 균형을 맞추는 문제다. 이 세 가지를 동시에 최대한 달성하기 어렵고, 특히 온디바이스 환경에서는 메모리 절감이 가장 큰 제약으로 작용한다. 따라서 이 보고서는 어떤 성능 요소를 포기했는지, 그리고 그 포기가 스마트폰 온디바이스 LLM 도메인에서 감당 가능한 수준인지에 초점을 맞춘다.

## 2. 기술 선정

**선정 기준** (설계서 1.1)

- **온디바이스 적용 가능성** — 별도의 학습이나 전용 하드웨어 없이 기존 모델에 적용할 수 있는가
- **평가 자료 확보 가능성** — 오픈소스 구현·후속 연구·개발자 논의·제품 적용 사례가 충분한가
- **기술 성숙도** — TRL 기준 성숙 수준이 비교 가능한가, 공개 시점 차이로 불공정하지 않은가
- **RAG Corpus 적합성** — KV cache 가 논문의 주제인가, 2편 합쳐 200p 이내인가

**선정 기술**

- **SW · KIVI** — KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (ICML 2024, PMLR 235, arXiv:2402.02750, 15p)
  - 사후·무보정 2bit KV 양자화. 재학습·보정 데이터 없이 배포된 모델에 즉시 적용 가능, 피크 메모리 2.6x 감소. 정확도 trade-off 가 관점 대조 재료 (설계서 1.3)
- **HW · InfiniGen** — InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management (OSDI 2024, arXiv:2406.19707, 18p)
  - 동적 KV 오프로딩·프리패치. 전용 HW 없이 메모리 계층을 쓰는 유일한 후보, 정확도 무손실 vs 전송·전력 비용 (설계서 1.4)

**검토했으나 제외한 후보**

- TurboQuant (SW) — 설계서 1.2 참고
- DeepSeek-V2 MLA (SW) — 아키텍처 재설계 — 배포된 모델에 사후 적용 불가
- ITME CXL (HW) — 전용 인터커넥트 필요 — 스마트폰에 없음
- PIM-CXL (HW) — 전용 HW 필요

## 3. 기술 개요

### KIVI

KIVI는 긴 컨텍스트나 배치 추론 시 KV 캐시의 저장 및 로딩에서 발생하는 메모리와 속도 병목 문제를 완화하기 위해 KV 캐시의 총 바이트 수를 줄이는 2비트 양자화 방식을 적용한다. KV 캐시를 그룹 단위로 양자화하고, 잔여 부분은 원래 정밀도로 유지하여 긴 시퀀스에서 메모리 오버헤드를 최소화한다. 또한, 디퀀타이즈와 행렬 곱셈을 융합하는 혼합 정밀도 연산으로 효율성을 높인다. KIVI는 Llama-2-7B 모델에서 KV 캐시를 2비트로 압축해 최대 2.6배 피크 메모리 사용량 감소, 최대 4배 배치 크기 지원, 2.35배에서 3.47배 처리량 향상을 보였다[p.2][p.3][p.5][p.8][p.9][p.12]. 한계로는 2비트 양자화가 multi-query attention과 같이 KV 캐시가 하나의 헤드만 사용되는 경우 정확도 하락이 발생할 수 있어, 4비트 양자화가 필요할 때가 있다[p.6]. KIVI는 메모리 절감과 처리량 향상을 얻는 대신 일부 상황에서 정확도 유지에 어려움이 있을 수 있다[추론].

### InfiniGen

InfiniGen은 긴 텍스트 생성에 최적화된 KV 캐시 관리 프레임워크로, 오프로딩 기반 추론 시스템과 연계하여 CPU 메모리의 KV 캐시 중 중요한 토큰만을 추측적으로 미리 GPU로 가져온다. 이전 레이어의 attention 입력과 쿼리 및 키 가중치를 활용해 필수 KV 캐시 항목을 동적으로 선별하고, 불필요한 항목은 제거하여 전송 오버헤드를 줄인다. 이를 통해 KV 캐시의 선형 확장 문제를 완화하고 추론 지연 시간을 단축한다. InfiniGen은 OPT 6.7B, 13B, 30B 및 Llama-2 7B, 13B 모델에서 평가되었으며, FlexGen 대비 데이터 전송량 감소로 추론 속도를 크게 향상시키고, Ideal 대비 1.52배 느린 반면 다른 방법들은 3.90배에서 18.55배 느린 결과를 보였다[p.1][p.2][p.4][p.6][p.9][p.13][p.14]. 한계로는 KV 캐시 크기가 10% 이하로 작아질 경우 정확도가 떨어질 수 있으며, KV 캐시를 CPU에서 GPU로 전송하는 과정이 새로운 병목이 될 수 있다고 지적된다[p.1][p.10]. InfiniGen은 추론 지연 시간 단축과 확장성을 얻는 대신, KV 캐시 전송 과정에서 병목 가능성이 존재한다[추론].

## 4. 관점별 평가

### 4.1 기술 성숙도 (TRL — 공개 정보 기반 추정, 기준 시점 명시)

- **KIVI** — TRL 4 (기준 시점: 2024 공개 자료 기준)
    - [논문 2402.02750] 공개 논문 및 GitHub 구현체 확인
    - [웹 https://github.com/jy-yuan/KIVI] 공개 코드 및 하드웨어 친화적 구현체 존재
- **InfiniGen** — TRL 3 (기준 시점: 2024 공개 자료 기준)
    - [논문 2406.19707] 공개 논문 및 GitHub 구현체 확인
    - 재학습·보정 데이터 및 벤더 제품 적용 발표 미확인 [추론]

TRL 4~6 구간은 수율·성능 수치가 비공개라 정보 공백이 크고, 발표 시점과 채택 사이에 시차가 있다. 위 등급은 공개 정보 기반 추정이다.

### 4.2 시장

- **KIVI** — 등급: 채택: 중 / 시장 연결: 중 / 생태계: 중
    - 근거: 채택: KIVI는 Llama, Falcon, Mistral 등 인기 모델 패밀리에 적용되어 평가되었으며, HuggingFace Transformers의 KV Cache 양자화에도 영감을 주었다. [웹 https://github.com/jy-yuan/KIVI]
시장 연결: KIVI는 KV 캐시 메모리 절감과 처리량 향상을 통해 온디바이스 LLM 추론 효율화를 목표로 하며, 관련 시장에서 메모리 절감 수요에 부합한다. [웹 https://arxiv.org/html/2402.02750v2]
생태계: KIVI는 CUDA 및 Triton을 이용한 하드웨어 친화적 구현체가 존재하며, GitHub에 코드가 공개되어 있다. [웹 https://github.com/jy-yuan/KIVI]
    - 긍정:
        - KIVI는 Llama-2-7B에서 KV 캐시를 2bit로 압축하여 2.6배 메모리 사용량 감소와 최대 4배 배치 크기 지원, 2.35~3.47배 처리량 향상을 달성했다. [웹 https://arxiv.org/html/2402.02750v2]
        - KIVI는 미세 조정 없이 하드웨어 친화적인 2bit KV 캐시 양자화 알고리즘으로, Llama, Falcon, Mistral 모델에서 평가되었다. [웹 https://arxiv.org/html/2402.02750v2]
        - KIVI는 CUDA와 Triton을 이용해 디퀀타이즈와 행렬 곱셈을 융합하는 효율적인 구현체를 제공한다. [웹 https://arxiv.org/html/2402.02750v2]
    - 부정:
        - 2bit KIVI는 multi-query attention과 KV cache에 하나의 헤드만 사용하는 경우 큰 정확도 하락이 발생할 수 있어 4bit KIVI가 필요할 수 있다. [웹 https://arxiv.org/html/2402.02750v2]
        - KIVI는 단일 NVIDIA A100 GPU (80GB) 환경에서 평가되어, 다양한 하드웨어 환경에서의 성능 검증은 제한적이다. [웹 https://arxiv.org/html/2402.02750v2]
    - confidence: 0.8

- **InfiniGen** — 등급: 채택: 근거 없음 / 시장 연결: 근거 없음 / 생태계: 상
    - 근거: 채택: 유효한 근거 없음 [추론]
시장 연결: 유효한 근거 없음 [추론]
생태계: InfiniGen은 OSDI 2024에서 발표되었고, GitHub에 구현체가 공개되어 있으며, 여러 LLM 모델에서 평가되었다. [웹 https://github.com/snu-comparch/InfiniGen]
검색 결과로 확인되지 않은 근거 제외 [추론]
    - 긍정:
        - 근거 없음
    - 부정:
        - CPU 메모리에서 GPU로 KV 캐시를 전송하는 과정이 LLM 추론에서 새로운 성능 병목 현상이 될 수 있다. [웹 https://arxiv.org/abs/2406.19707]
    - confidence: 0.3


### 4.3 이해관계자 (찬 · 반)

- **KIVI** — 등급: 경쟁 기술 진영: 중립 / 도입 기업·개발자: 우호 / 투자·미디어: 근거 없음
    - 근거: 경쟁 기술 진영: TurboQuant와 같은 경쟁 기술은 KIVI와 달리 3~4비트 범위에서 낮은 왜곡과 장기 컨텍스트 품질 유지에 중점을 둔다. TurboQuant는 더 복잡한 수학적 기법으로 왜곡을 줄이는 반면, KIVI는 2비트 캐시 압축에 더 단순하고 직접적으로 최적화되어 있다. 따라서 두 기술은 정확도, 복잡성, 압축률 측면에서 서로 다른 최적점을 추구한다. [웹 https://turbo-quant.com/turboquant-vs-kivi]
도입 기업·개발자: KIVI는 미세 조정 없이 바로 적용 가능한 2비트 KV 캐시 양자화 알고리즘으로, Llama, Mistral, Falcon 등 여러 인기 모델에서 평가되었으며, 최대 2.6배 메모리 사용량 감소와 2.35~3.47배 처리량 향상을 달성했다. 하드웨어 친화적인 CUDA 및 Triton 구현을 제공하며, 별도의 재학습이나 보정 데이터가 필요하지 않아 도입 장벽이 낮다. [웹 https://arxiv.org/html/2402.02750v2]
투자·미디어: 직접적인 투자자나 미디어의 반응이나 사업성 평가 자료가 제공되지 않아 판단 근거가 부족하다. [추론]
    - 긍정:
        - KIVI는 Llama-2-7B 모델에서 KV 캐시를 2비트로 압축하여 2.6배의 피크 메모리 사용량 감소를 달성하고, 최대 4배 더 큰 배치 크기와 2.35~3.47배 처리량 향상을 보였다. [웹 https://arxiv.org/html/2402.02750v2]
        - KIVI는 키 캐시는 채널 단위로, 값 캐시는 토큰 단위로 비대칭적으로 양자화하여 2비트 극한 압축에도 불구하고 정확도 손실을 최소화한다. [웹 https://liner.com/review/kivi-tuningfree-asymmetric-2bit-quantization-for-kv-cache]
        - KIVI는 미세 조정 없이 바로 적용 가능한 플러그 앤 플레이 방식이며, CUDA와 Triton을 이용한 하드웨어 친화적 구현으로 실제 GPU 환경에서 효율적인 연산을 지원한다. [웹 https://arxiv.org/html/2402.02750v2]
        - KIVI는 Llama, Falcon, Mistral 등 다양한 인기 모델에 적용되어 거의 동일한 품질을 유지하면서 메모리 사용량을 크게 줄이고 처리량을 향상시켰다. [웹 https://medium.com/@michael.hannecke/googles-turboquant-changes-the-economics-of-local-ai-inference-acce5839014d]
    - 부정:
        - KIVI 2비트 양자화는 경우에 따라 큰 정확도 하락을 보일 수 있으며, 특히 multi-query attention과 KV 캐시에 하나의 헤드만 사용하는 경우 4비트 KIVI가 필요하다. (원문: ## 2 Background: Attention Inference-Time Workflow [...] Thus, in Table 3, 4bit KIVI is needed to maintain the accuracy, while 2bit KIVI may have a large accuracy drop in this case.) [웹 https://arxiv.org/html/2402.02750v2]
        - KIVI는 2비트 양자화의 정확도 하락 문제를 완전히 해결하지 못해 일부 복잡한 추론 작업에서는 정확도 손실이 발생할 수 있다. (원문: For instance, on Llama-2-7B, KIVI-2 shows only a minor accuracy drop (e.g., 13.50 to 12.74 on GSM8K) compared to the severe degradation seen with other 2-bit configurations (e.g., 0.83 for K-T, V-T on GSM8K).) [웹 https://liner.com/review/kivi-tuningfree-asymmetric-2bit-quantization-for-kv-cache]
    - confidence: 0.8

- **InfiniGen** — 등급: 경쟁 기술 진영: 중립 / 도입 기업·개발자: 우호 / 투자·미디어: 근거 없음
    - 근거: 경쟁 기술 진영: InfiniGen은 기존 KV 캐시 관리 방법들에 비해 데이터 전송량 감소와 추론 속도 향상에서 우수한 성능을 보이나, H2O와 같은 다른 접근법은 메모리 사용량 감소와 처리량 측면에서 장점이 있다. 다만 H2O는 정확도 저하가 발생하는 반면 InfiniGen은 정확도 보존에 강점이 있다. [웹 https://arxiv.org/pdf/2604.05012]
도입 기업·개발자: InfiniGen은 긴 텍스트 생성에 최적화된 KV 캐시 관리 프레임워크로, 기존 오프로딩 기반 추론 시스템과 시너지 효과를 내며 작동한다. 다양한 대형 언어 모델에서 추론 지연 시간을 크게 단축하고 모델 성능을 유지하며, 배치 크기, 시퀀스 길이, 모델 크기에 대해 이전 솔루션보다 더 나은 확장성을 보였다. [웹 https://arxiv.org/html/2406.19707v1]
투자·미디어: 투자자 및 미디어의 직접적인 반응이나 평가 자료가 제공되지 않아 판단 근거가 부족하다. [추론]
    - 긍정:
        - InfiniGen은 기존 FlexGen 대비 데이터 전송량 감소로 추론 속도를 크게 향상시켰으며, Ideal 대비 1.52배 느린 반면 다른 방법들은 3.90배에서 18.55배 느리다. [웹 https://arxiv.org/html/2406.19707v1]
        - InfiniGen은 긴 텍스트 생성에 최적화된 새로운 KV 캐시 관리 프레임워크로, 오프로딩 기반 추론 시스템과 시너지 효과를 내며 작동한다. [웹 https://arxiv.org/html/2406.19707v1]
        - InfiniGen은 추론 지연 시간을 크게 단축하면서 언어 모델 성능을 유지하였고, 배치 크기, 시퀀스 길이, 모델 크기에 대해 이전 솔루션보다 더 나은 확장성을 보였다. [웹 https://arxiv.org/html/2406.19707v1]
        - InfiniGen은 중요한 토큰의 KV 캐시를 추측적으로 미리 가져와 GPU로 전송하는 데이터 양을 줄여 KV 캐시 전송 오버헤드를 크게 감소시킨다. [웹 https://arxiv.org/html/2406.19707v1]
        - InfiniGen은 여러 대표적인 대형 언어 모델에서 기존 KV 캐시 관리 방법 대비 최대 3배의 성능 향상을 보이며, 모델 정확도도 상당히 우수하다. [웹 https://arxiv.org/html/2406.19707v1]
    - 부정:
        - KV 캐시 크기가 10%보다 작을 때 InfiniGen의 정확도가 눈에 띄게 떨어지는 경우가 있으며, 이는 비트 폭이 충분하지 않은 양자화나 영구적인 KV 캐시 제거 때문일 수 있다. (원문: For relative KV cache sizes larger than 10%, the accuracy with InfiniGen closely) [웹 https://arxiv.org/html/2406.19707v1]
        - CPU 메모리에서 GPU로 KV 캐시를 전송하는 과정이 LLM 추론에서 새로운 성능 병목 현상이 될 수 있다. (원문: Even so, InfiniGen shows a 1.34×\times speedup over FlexGen, while others) [웹 https://arxiv.org/html/2406.19707v1]
        - InfiniGen은 CPU-GPU 간 데이터 전송으로 인한 처리량 저하가 있어 고처리량 시나리오에는 제한적일 수 있다. (원문: CONCLUSION We compare three representative KV cache management frameworks: vLLM (memory management), H2O (static spar-sification), and InfiniGen (dynamic selection).) [웹 https://arxiv.org/pdf/2604.05012]
        - InfiniGen은 지연 시간 제약이 완화된 환경에서 정확도 저하를 최소화하는 희소화가 필요한 배포에 적합하지만, CPU-GPU 전송으로 인한 처리량 페널티가 존재한다. (원문: InfiniGen occupies a special-ized niche: deployments requiring sparsification with minimal accuracy degradation, where latency constraints are relaxed.) [웹 https://arxiv.org/pdf/2604.05012]
    - confidence: 0.85


### 4.4 도메인 — 스마트폰 온디바이스 (적합 / 조건부 / 부적합 + 포기한 축)

- **KIVI** — 등급: 적합
    - 판정: 적합
    - 근거: KIVI는 재학습이나 파인튜닝 없이 plug-and-play 방식으로 동작하며, 추가 하드웨어나 메모리 계층, 대역폭 전제를 요구하지 않는다[논문 p.2]. 따라서 스마트폰 온디바이스 LLM 배포 제약에 부합한다. 정확도 손실과 오버헤드는 판정 조건이 아니지만, INT2 양자화 시 정확도 저하가 있으나 KIVI는 INT2가 아닌 2bit 비대칭 양자화를 사용하여 큰 손실 없이 메모리 절감 효과를 달성한다[논문 p.3][논문 p.8].
    - 긍정:
        - 재학습·튜닝·보정 데이터가 필요 없으며 plug-and-play 방식으로 적용 가능하다[논문 p.2]
        - 추가 하드웨어, 메모리 계층, 대역폭 전제 조건이 논문에 근거 없음으로 확인되어 대상 기기 자원 전제를 하지 않는다[논문 p.2]
    - 부정:
        - INT2 양자화 시 정확도에 눈에 띄는 하락이 발생한다[논문 p.3]
        - value cache를 per-channel 양자화하면 정확도가 크게 악화된다[논문 p.3]
    - 3축: recall — INT2 양자화 시 정확도 저하가 있으나 KIVI는 2bit 비대칭 양자화로 정확도 손실을 최소화함[논문 p.3] · latency — 논문에 명시된 처리량 증가(2.35×∼3.47×)는 지연 감소에 긍정적 영향을 미침[논문 p.2] · memory — KV cache 메모리 사용량을 2.6× 절감하여 메모리 부담을 크게 줄임[논문 p.2]
    - confidence: 0.9

- **InfiniGen** — 등급: 조건부
    - 판정: 조건부
    - 근거: InfiniGen은 KV 캐시를 GPU 메모리 대신 CPU 메모리에 오프로딩하는 방식을 전제로 하며, 이는 스마트폰 온디바이스 LLM의 메모리 제약과 배포 제약에 부합하지 않는다[논문 p.4][논문 p.6]. 또한, 재학습이나 파인튜닝 없이 적용 가능하다는 근거는 논문에 명확히 제시되어 있지 않다[논문에 근거 없음]. 따라서 대상 기기에 없는 자원(CPU 메모리 오프로딩)을 전제로 하지만, 스마트폰 환경에서 CPU 메모리 활용 및 PCIe 대역폭 관리로 우회 가능성이 있으므로 조건부 적합으로 판단한다[추론].
    - 긍정:
        - 정확도 손실에 대한 실험 조건과 수치가 논문에 일부 제시되어 있어, 성능 저하를 최소화하는 방법을 확인할 수 있다[논문 p.9][논문 p.12].
        - 추가 하드웨어 없이 재학습·튜닝이 필요 없다는 점은 논문에 근거 없음으로 명확하지 않으나, 오프로딩 방식으로 메모리 부담을 줄이는 시도는 긍정적이다[논문에 근거 없음][논문 p.6].
    - 부정:
        - InfiniGen은 CPU 메모리 오프로딩과 PCIe 대역폭 관리를 전제로 하여, 스마트폰 온디바이스 LLM의 메모리 및 배포 제약에 부합하지 않는다[논문 p.4][논문 p.6].
        - 재학습·튜닝·보정 데이터가 필요하다는 근거는 없으나, 논문에서 명확히 부정하지 않아 배포 제약 충족 여부가 불확실하다[논문에 근거 없음].
        - 데이터 전송 및 연산 오버헤드가 Transformer 실행 시간의 대부분을 차지하여, 추론 지연(latency)이 증가할 가능성이 크다[논문 p.4][논문 p.13].
    - 3축: recall — 정확도 손실 수치는 논문에서 일부 제시되었으나, 전체적인 정보 보존 정도는 불확실하다[논문 p.9][논문 p.12]. · latency — PCIe 대역폭 제한과 데이터 전송 오버헤드로 인해 추론 지연이 크게 증가할 수 있다[논문 p.4][논문 p.13]. · memory — KV 캐시를 CPU 메모리에 오프로딩하여 GPU 메모리 사용량을 줄이는 효과가 있으나, 스마트폰 메모리 환경과는 차이가 있다[논문 p.6].
    - confidence: 0.85

## 5. 시사점

KIVI는 온디바이스 스마트폰 환경에 적합한 plug-and-play 2bit 양자화 기술로 평가된다. 이는 추가 하드웨어나 메모리 계층, 대역폭 전제를 요구하지 않고, 재학습이나 보정 데이터 없이 동작하는 점에서 스마트폰 배포 제약에 부합하는 것으로 해석된다. 반면 InfiniGen은 CPU 메모리 오프로딩과 PCIe 대역폭 관리가 전제되어 있어, 스마트폰 온디바이스 LLM 배포에 적합하지 않다는 평가가 나온다. 이러한 차이는 두 기술이 요구하는 하드웨어 및 시스템 환경의 차이에서 비롯된다[논문 2402.02750 p.2, 2406.19707 p.4, p.6]. 판단에 필요한 추가 확인 항목은 스마트폰 환경에서 PCIe 대역폭과 메모리 오프로딩이 실제로 추론 지연에 미치는 영향이다.

시장 및 채택 관점에서는 KIVI가 Llama, Falcon, Mistral 등 인기 모델 패밀리에 적용되어 평가되었고, HuggingFace Transformers의 KV Cache 양자화에 영향을 준 점이 중간 수준의 채택 근거로 제시된다. 반면 InfiniGen은 구현체가 공개되어 있으나 채택과 시장 연결에 대한 유효한 근거가 부족하여 평가가 엇갈린다. 이 차이는 공개된 적용 사례와 시장 반응의 유무에서 기인한다[추론]. 판단에 필요한 추가 확인 항목은 InfiniGen의 실제 산업 현장 적용 사례 및 시장 반응 데이터다.

기술 성숙도(TRL) 평가에서는 KIVI가 공개 논문과 구현체가 모두 확인되어 TRL 4로 평가된다. InfiniGen은 공개 논문과 구현체만 확인되었으며, 재학습·보정 데이터 및 벤더 제품 적용 발표가 미확인되어 TRL 3으로 평가된다. 이 차이는 기술 검증 범위와 적용 증거의 차이에 따른 것이다[추론]. 판단에 필요한 추가 확인 항목은 InfiniGen의 재학습·보정 데이터 존재 여부와 벤더 제품 적용 발표의 확인이다.

이해관계자 관점에서는 KIVI가 경쟁 기술 대비 단순하고 직접적인 2bit 압축 기술로 도입 기업과 개발자에게 긍정적으로 평가된다. 반면 InfiniGen은 데이터 전송 병목과 정확도 저하 가능성으로 인해 제한적인 평가를 받는다. 이러한 엇갈림은 각 기술이 제공하는 성능 개선과 구현 복잡성, 정확도 유지 측면에서 이해관계자가 중시하는 요소가 다르기 때문이다[추론]. 판단에 필요한 추가 확인 항목은 InfiniGen의 데이터 전송 병목 현상과 정확도 저하 영향에 대한 실측 결과다.

두 기술 모두 재학습이나 보정 데이터 없이 기존 모델에 적용 가능하다는 점, KV 캐시 관리의 중요성을 인지하고 메모리 사용량 및 처리량 개선을 목표로 한다는 점, 그리고 여러 인기 LLM 모델에서 평가되어 GPU 환경에서 효율성 개선을 추구한다는 점에서는 일치한다[논문 2402.02750 p.2, 2406.19707 p.1, p.9].

## 6. 한계점

1. **공개 정보 기반 추정의 한계** — TRL 4~6 구간은 수율·성능 수치가 비공개라 정보 공백이 가장 크다. 4.1 의 등급은 공개 코드·재현·프레임워크 통합 여부만으로 추정한 것이다.
2. **TRL 기준 시점** — arXiv v1 과 학회 게재·가이드 표기 시점이 논문마다 다르다(KIVI: v1 2024-02 / ICML 2024-07, InfiniGen: v1 2024-06 / OSDI 2024-07). 기준 시점에 따라 추정이 달라진다 — 발표 시점과 채택 간 시차의 구체 사례다.
3. **사전 가설과 확증편향 방지 조치** — 사전 가설 *"온디바이스에서는 SW 압축이 더 적합할 것"* 을 명시하고 장치 8개(기술별 독립 호출 · 사실 단위 질의 · 정확도 임계값 없음 · 근거 없음 기록 · 출처 태그 강제 · 반대 근거 ≥2 · 중립성 검증 루프)를 적용했다. 판정 기준(재학습 불필요 · 전용 HW 불필요)은 배포 용이성에 가중을 두므로 구조적으로 SW 접근에 유리하다 — 도메인의 실제 제약을 반영한 것이지만 기준 선택 자체가 결과에 영향을 준다. Memory · Recall · Latency 3축은 판정이 아닌 해석에만 썼다.
4. **`[추론]` 태그 비율** — 판단 문장 145건 중 논문·웹 근거 없이 추론에만 의존한 문장 5% (7/145).
5. **검색·생성 품질** — Hit Rate@4 0.8 · MRR@4 0.52 (검색 구성 `dual-bm25/ko(dense)+en(sparse)`, 20문항) · RAGAS Faithfulness 0.936 · ResponseRelevancy 0.838 · ContextPrecision 0.912. 검색 20회 중 "논문에 근거 없음" 4건(KIVI 2건 · InfiniGen 2건). 임베딩 비교는 20문항 기준이라 0.10 차이는 2문항이며, 선정은 수치 우위가 아니라 한국어 질의 요건·컨텍스트 길이에 둔다. 근거 없음이 한 기술에 몰리면 그 기술 판정의 `[추론]` 비중이 높아진다. 웹 출처는 Tavily 가 저자·게시일을 주지 않는 경우가 많아 REFERENCE 에 기관명(사이트)과 접근일로 대체했다 — 게시일이 필요한 항목은 사람이 확인해야 한다.
6. **질의 재작성 효과** — 재작성 9건 중 5건이 관련 문서를 찾았다. 효과가 없으면 재작성 단계 제거를 검토한다.

## REFERENCE

- Liu, Z., Yuan, J., Jin, H. et al.(2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. *Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235*. arXiv:2402.02750.
- Lee, W., Lee, J., Seo, J., Sim, J.(2024). InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*. arXiv:2406.19707.
- jy-yuan(게시일 미상 · 2026-09-22 접근). *[ICML 2024] KIVI: A Tuning-Free Asymmetric 2bit ...*. github.com, https://github.com/jy-yuan/KIVI
- arxiv.org(게시일 미상 · 2026-09-22 접근). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. arxiv.org, https://arxiv.org/html/2402.02750v2
- arxiv.org(게시일 미상 · 2026-09-22 접근). *InfiniGen: Efficient Generative Inference of Large ...*. arxiv.org, https://arxiv.org/abs/2406.19707
- snu-comparch(게시일 미상 · 2026-09-22 접근). *GitHub - snu-comparch/InfiniGen: InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management (OSDI'24) · GitHub*. github.com, https://github.com/snu-comparch/InfiniGen
- turbo-quant.com(게시일 미상 · 2026-09-22 접근). *TurboQuant vs KIVI: KV Cache Quantization Compared | TurboQuant Tools*. turbo-quant.com, https://turbo-quant.com/turboquant-vs-kivi
- medium.com(게시일 미상 · 2026-09-22 접근). *TurboQuant Changes the Economics of Local AI Inference - Medium*. medium.com, https://medium.com/@michael.hannecke/googles-turboquant-changes-the-economics-of-local-ai-inference-acce5839014d
- liner.com(게시일 미상 · 2026-09-22 접근). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache [Quick Review]*. liner.com, https://liner.com/review/kivi-tuningfree-asymmetric-2bit-quantization-for-kv-cache
- arxiv.org(게시일 미상 · 2026-09-22 접근). *InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management*. arxiv.org, https://arxiv.org/html/2406.19707v1
- arxiv.org(게시일 미상 · 2026-09-22 접근). *Comparative Characterization of KV Cache Management ...*. arxiv.org, https://arxiv.org/pdf/2604.05012
