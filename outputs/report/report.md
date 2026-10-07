# KV cache 최적화 기술 다관점 평가 — KIVI(SW) · InfiniGen(HW) · 스마트폰 온디바이스 LLM

## SUMMARY

KIVI와 InfiniGen 두 KV cache 최적화 기술을 스마트폰 온디바이스 LLM 도메인에서 기술 성숙도, 시장, 이해관계자, 도메인 적합성 관점으로 비교했다. 기술 성숙도는 두 기술 모두 TRL 4로 공개 정보 기반 추정되며, 시장에서는 KIVI가 메모리 절감과 처리량 향상 중심, InfiniGen은 데이터 전송 최적화와 추론 속도 개선 중심으로 평가된다. 이해관계자 관점에서는 KIVI가 개발자 중심 우호적 반응을, InfiniGen은 투자·미디어에서 우호적 평가를 받지만 정확도 손실 우려가 존재한다. 도메인 적합성에서는 KIVI가 메모리 절감 효과에도 온디바이스 메모리 대역폭과 전력 제한 문제로 조건부 적합, InfiniGen은 CPU 메모리 오프로딩과 PCIe 대역폭 의존으로 스마트폰 환경 적합성이 불확실하다. 가장 큰 엇갈림은 KIVI가 압축 효율과 하드웨어 친화성에 초점을 맞추는 반면, InfiniGen은 시스템 레벨 전송 최적화에 집중하는 점과, 두 기술 모두 스마트폰 온디바이스 환경에서 실험적 검증 부족으로 실제 성능과 정확도 유지에 대한 판단이 어렵다는 점이다. 본 보고서 평가는 공개 정보 기반 추정이며, 판단 문장 중 약 5%가 추론에 의존한다[추론].

## 1. 분석 배경

KV cache는 토큰당 약 0.5MB(7B 모델, fp16, MHA 기준) 크기로, 문맥 길이에 비례해 용량이 증가한다. 이로 인해 초기에는 연산 병목이었으나, 문맥이 길어질수록 메모리 병목으로 전환되는 특성이 있다[추론]. KV cache 최적화 접근법은 크게 두 가지로 나뉜다. 첫째는 소프트웨어적 방법으로, 데이터를 압축하거나 양자화하여 크기를 줄이는 방식이다. 둘째는 하드웨어적 방법으로, 메모리나 스토리지 계층을 확장해 담을 공간을 넓히는 방식이다[추론].

스마트폰 온디바이스 LLM 도메인은 메모리 확장이 불가능하고 배터리 제약이 있으며, 재학습이 불가한 환경이다. 또한 어시스턴트 응답 품질이 제품 품질과 직결되므로 압축 손실에 민감하다. 대표적인 시나리오로는 RAM 12GB 환경에서 4bit 7B 모델을 사용해 8K 토큰 문맥을 처리할 때, fp16 KV cache가 약 4GB를 차지하는 상황이 있다[domain.constraints, domain.scenario]. 이처럼 온디바이스 환경에서는 메모리가 절대적인 제약 요소로 작용한다.

성능 평가 축은 Recall, Latency, Memory 세 가지로 정의된다. 이 세 축을 동시에 최적화하기는 어려우며, 특히 온디바이스 환경에서는 Memory 제약이 가장 큰 영향을 미친다. 따라서 이 보고서는 각 기술이 어떤 성능 축을 희생했는지, 그리고 그 희생이 스마트폰 온디바이스 LLM 도메인에서 감당 가능한 수준인지에 초점을 맞춘다[domain.axes, 추론].

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
KIVI는 KV 캐시를 2bit로 극저비트 양자화하여 메모리 사용량을 줄이고 배치 크기와 처리량을 높이는 데 초점을 맞춘 기술이다[p.2][p.3][p.9]. key cache는 per-channel 단위로, value cache는 per-token 단위로 양자화하여 각 캐시의 분포 특성에 맞게 오류를 국한시키며, 스트리밍 데이터 구조를 활용해 효율적으로 처리한다[p.2][p.3][p.9]. KIVI는 Llama-2-7B 모델에서 KV 캐시를 2bit로 압축해 2.6× 피크 메모리 사용량 감소, 최대 4× 배치 크기 지원, 2.35×~3.47× 처리량 향상을 보였다. 실험은 단일 NVIDIA A100 GPU (80GB) 환경에서 수행되었다[p.2][p.1][p.8]. 한계로는 2bit 양자화 시 4bit 대비 정확도 하락이 크고, 특히 Llama2-7B 모델에서 정확도에 영향이 있으며, residual key/value cache는 최대 128 토큰까지 full precision으로 유지해 긴 컨텍스트에서 메모리 오버헤드가 발생할 수 있다[p.6][p.9][p.5]. KIVI는 메모리 사용량을 줄이고 처리량을 높이는 대신, 일부 정확도와 긴 컨텍스트 처리 시 메모리 오버헤드를 내주는 구조로 보인다[추론].  

### InfiniGen  
InfiniGen은 오프로딩 기반 추론 시스템과 연동해 Transformer의 다음 어텐션 레이어 계산에 필요한 핵심 토큰을 현재 레이어 입력과 다음 레이어 쿼리 가중치 및 키 캐시 일부를 이용해 추측하고, 필요한 KV 캐시만 GPU로 사전 가져오기(prefetch)하는 방식으로 데이터 전송량을 줄이는 KV 캐시 관리 프레임워크이다[p.1][p.2][p.4]. 또한, 사용 빈도가 낮은 토큰의 KV 캐시 항목을 동적으로 제거하며 FIFO, LRU, 카운터 기반 정책을 적용해 캐시 풀을 관리한다[p.1][p.2][p.9][p.14]. InfiniGen은 Open Pre-trained Transformer (OPT) 6.7B, 13B, 30B와 Llama-2 7B, 13B 모델에 적용되었으며, FlexGen 대비 데이터 전송량 감소로 추론 속도를 크게 향상시켰다. Ideal 대비 1.52× 느리지만 다른 방법들은 3.90×~18.55× 느리다[p.13][p.9][p.14]. 한계로는 KV cache 크기가 10% 이하일 때 정확도가 눈에 띄게 떨어지며, 이는 비트 폭 부족 양자화나 영구적 캐시 제거 때문이고, CPU에서 GPU로의 KV 캐시 전송이 새로운 병목 현상이 될 수 있음을 지적한다[p.10][p.1]. InfiniGen은 데이터 전송량과 추론 지연 시간을 줄이는 데 집중하면서, 일부 정확도 저하 위험과 전송 병목 문제를 내포하는 구조로 평가된다[추론].

## 4. 관점별 평가

### 4.1 기술 성숙도 (TRL — 공개 정보 기반 추정, 기준 시점 명시)

- **KIVI** — TRL 4 (기준 시점: 2024 공개 자료 기준)
    - 논문 2402.02750: 공개 코드 및 GPU 친화적 구현(CUDA, Triton) 확인 [추론]
    - ICML 2024 발표 [추론]
- **InfiniGen** — TRL 4 (기준 시점: 2024 공개 자료 기준)
    - 논문 2406.19707: 구현체 및 평가 스크립트 공개 확인 [추론]
    - USENIX OSDI 2024 발표 [추론]

TRL 4~6 구간은 수율·성능 수치가 비공개라 정보 공백이 크고, 발표 시점과 채택 사이에 시차가 있다. 위 등급은 공개 정보 기반 추정이다.

### 4.2 시장

- **KIVI** — 등급: 채택: 상 / 시장 연결: 중 / 생태계: 상
    - 긍정: KIVI는 2bit KV 캐시 양자화를 통해 Llama-2-7B 모델에서 2.6배 메모리 사용량 감소와 최대 4배 배치 크기 지원, 2.35~3.47배 처리량 향상을 달성했다 [웹 https://github.com/jy-yuan/KIVI]. / KIVI는 별도의 재학습이나 보정 없이 GPU에서 CUDA와 Triton을 활용한 하드웨어 친화적 구현으로 즉시 적용 가능하다 [웹 https://github.com/jy-yuan/KIVI].
    - 부정: 2bit KIVI는 4bit KIVI에 비해 정확도 하락이 크며, 특히 Llama2-7B 모델에서 정확도에 큰 영향을 미친다 [웹 https://github.com/jy-yuan/KIVI]. / KV 캐시의 residual key/value cache는 최대 128 토큰까지 full precision으로 유지해야 하며, 긴 컨텍스트 시나리오에서 메모리 오버헤드가 발생할 수 있다 [웹 https://github.com/jy-yuan/KIVI].

- **InfiniGen** — 등급: 채택: 중 / 시장 연결: 중 / 생태계: 중
    - 긍정: InfiniGen은 기존 FlexGen 대비 데이터 전송량 감소로 추론 속도를 크게 향상시켰다 [웹 https://huggingface.co/papers/2406.19707]. / OPT 및 Llama-2 등 다양한 대형 언어 모델에 적용되어 성능과 정확도를 유지하면서 추론 지연 시간을 크게 단축한다 [웹 https://huggingface.co/papers/2406.19707].
    - 부정: KV 캐시 크기가 10% 이하로 줄어들면 정확도가 눈에 띄게 떨어지는 제약이 있다 [웹 https://huggingface.co/papers/2406.19707]. / KV 캐시를 CPU에서 GPU로 전송하는 과정이 LLM 추론의 새로운 성능 병목 현상이 될 수 있다 [웹 https://huggingface.co/papers/2406.19707].


### 4.3 이해관계자 (찬 · 반)

- **KIVI** — 등급: 경쟁 기술 진영: 중립 / 도입 기업·개발자: 우호 / 투자·미디어: 중립
    - 긍정: KIVI는 Llama-2-7B 모델에서 KV 캐시를 2비트로 압축하여 2.6배의 피크 메모리 사용량 감소를 달성하고, 최대 4배 더 큰 배치 크기를 지원하며 2.35배에서 3.47배 사이의 처리량 향상을 보였다 [웹 https://www.alphaxiv.org/abs/2402.02750]. / KIVI는 튜닝이 필요 없는 2비트 비대칭 KV 캐시 양자화 방법으로, Llama, Falcon, Mistral 모델에서 거의 동일한 품질을 유지하면서 메모리 사용량을 크게 줄이고 처리량을 증가시킨다 [웹 https://turbo-quant.com/ko/turboquant-vs-kivi].
    - 부정: 2비트 KIVI는 4비트 KIVI에 비해 정확도 하락이 크며, 특히 Llama2-7B 모델에서 2비트 KIVI는 정확도에 큰 영향을 미친다 [웹 https://www.alphaxiv.org/abs/2402.02750]. / 긴 컨텍스트 시나리오에서 residual key cache와 value cache를 최대 128 토큰까지 full precision으로 유지해야 하므로, 이로 인한 메모리 오버헤드가 존재한다 [웹 https://www.alphaxiv.org/abs/2402.02750].

- **InfiniGen** — 등급: 경쟁 기술 진영: 중립 / 도입 기업·개발자: 우호 / 투자·미디어: 우호
    - 긍정: InfiniGen은 데이터 전송량 감소로 FlexGen 대비 추론 속도를 크게 향상시켰으며, Ideal 대비 1.52배 느리지만 다른 방법들은 3.90배에서 18.55배 느리다 [웹 https://huggingface.co/papers/2406.19707]. / InfiniGen은 배치 크기, 시퀀스 길이, 모델 크기 측면에서 이전 솔루션보다 더 나은 확장성을 보이고 추론 지연 시간을 크게 단축하면서 언어 모델 성능을 유지한다 [웹 https://huggingface.co/papers/2406.19707].
    - 부정: KV cache 크기가 10%보다 작을 때 정확도가 눈에 띄게 떨어지는 경우가 있으며, 이는 비트 폭이 부족한 양자화나 영구적인 KV cache 제거 때문임을 논문에서 언급한다 [웹 https://huggingface.co/papers/2406.19707]. / KV cache를 CPU 메모리에서 GPU로 전송하는 과정이 LLM 추론에서 새로운 성능 병목 현상이 될 수 있다 [웹 https://huggingface.co/papers/2406.19707].


### 4.4 도메인 — 스마트폰 온디바이스 (적합 / 조건부 / 부적합 + 포기한 축)

- **KIVI** — 등급: 조건부
    - 판정: 조건부
    - 긍정: fine-tuning 없이 plug-and-play 방식으로 적용 가능 [논문 p.2] / 2bit 압축으로 2.6배 메모리 사용량 절감 달성 [논문 p.2]
    - 부정: 온디바이스 AI 환경에서는 메모리 접근 지연과 전력 소모가 주요 병목으로, 단순한 메모리 압축만으로는 성능 보장이 어려움 [웹 https://ckhome7108.tistory.com/entry/%EC%98%A8%EB%94%94%EB%B0%94%EC%9D%B4%EC%8A%A4-AI%EA%B0%80-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EA%B5%AC%EC%A1%B0%EC%97%90-%EB%AF%B8%EC%B9%98%EB%8A%94-%EC%98%81%ED%96%A5] / 메모리 대역폭과 전력 제한으로 인해 스마트폰 온디바이스 환경에서 KIVI 구현 시 추가적인 최적화나 하드웨어 지원이 필요할 가능성 있음 [웹 https://m.blog.naver.com/PostView.naver?blogId=vitamin2230&logNo=224395215402]
    - 3축: recall — INT4 수준에서는 정확도 유지, INT2 수준에서는 정확도 하락 가능성 있음 [논문 p.3] · latency — 온디바이스 AI 환경에서 메모리 접근 지연과 전력 소모가 주요 병목으로, 추가 연산 및 데이터 이동 지연 가능성 있음 [웹 https://ckhome7108.tistory.com/entry/%EC%98%A8%EB%94%94%EB%B0%94%EC%9D%B4%EC%8A%A4-AI%EA%B0%80-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EA%B5%AC%EC%A1%B0%EC%97%90-%EB%AF%B8%EC%B9%98%EB%8A%94-%EC%98%81%ED%96%A5] · memory — KV cache 메모리 사용량을 2.6배 절감하여 메모리 부담 경감 [논문 p.2]

- **InfiniGen** — 등급: 조건부
    - 판정: 조건부
    - 긍정: InfiniGen는 재학습·튜닝·보정 데이터가 필요 없으며, GPU 메모리 부족 문제를 CPU 메모리 오프로드로 해결한다는 점에서 배포 제약에 부합한다 [논문 p.6][논문 p.9]. / KV cache 풀 크기를 관리하여 CPU 메모리 과부하를 방지하며, 필요한 PCIe 대역폭만 사용하여 대역폭 낭비를 줄인다 [논문 p.6][논문 p.9][논문 p.4].
    - 부정: InfiniGen는 PCIe 대역폭과 CPU 메모리 용량에 의존하는데, 스마트폰의 제한된 메모리 및 전력 환경과 차이가 있어 완전한 적합성은 불확실하다 [추론]. / 스마트폰 온디바이스 환경에서의 CPU 메모리 및 PCIe 대역폭 제약에 대한 실험적 평가가 논문에 충분히 이루어지지 않아, 실제 배포 적합성 판단에 한계가 있다 [논문 p.6][논문 p.9][논문 p.4].
    - 3축: recall — InfiniGen는 KV cache를 부분적으로 로드하여 원본 모델과 유사한 출력 품질을 유지하며, 정확도 손실은 상대 KV cache 크기 10% 미만에서 우수하다 [논문 p.9][논문 p.10][추론]. · latency — KV cache 전송 및 프리패치 오버헤드가 존재하지만, InfiniGen는 최소한의 키와 값 전송으로 PCIe 대역폭 낭비를 줄여 지연을 감소시킨다 [논문 p.4][논문 p.6][논문 p.9]. · memory — KV cache를 CPU 메모리에 오프로드하여 GPU 메모리 사용량을 줄이며, CPU 메모리 부담을 관리한다 [논문 p.6][논문 p.4].

## 5. 시사점

KIVI는 GPU 기반 2bit 양자화 기술로, 하드웨어 친화적 구현을 통해 온디바이스 적용 가능성을 제시한다. 그러나 스마트폰 환경에서의 메모리 대역폭과 전력 제한 문제는 여전히 해결 과제로 남아 있어, 조건부 적합성으로 평가된다. 반면 InfiniGen은 CPU 메모리 오프로딩과 PCIe 대역폭에 의존하는 설계로, 스마트폰의 제한된 메모리 및 전력 환경과 차이가 있어 완전한 적합성 여부가 불확실하다. 이로 인해 두 기술 모두 스마트폰 온디바이스 환경에서의 실험적 검증이 부족한 상황이다. 판단에 필요한 추가 확인 항목은 스마트폰 메모리 대역폭과 전력 제약 하에서의 실제 성능 및 전송 지연 실측이다[추론].

시장과 이해관계자 관점에서는 KIVI가 높은 메모리 절감과 처리량 향상을 중심으로 개발자와 투자자에게 긍정적 반응을 얻고 있다. 반면 InfiniGen은 데이터 전송 최적화와 추론 속도 개선을 강조하며 투자·미디어에서 우호적 평가를 받지만, 정확도 손실과 경쟁 기술 대비 한계가 지적되는 점이 엇갈린 평가를 낳는다. 이러한 차이는 KIVI가 압축 효율과 하드웨어 친화성에 초점을 맞춘 반면, InfiniGen은 시스템 레벨 전송 최적화에 집중한 데서 기인한다. 판단에 필요한 추가 확인 항목은 두 기술의 실제 추론 정확도 변화와 사용자 체감 성능 비교이다[추론].

정확도 및 성능 한계 측면에서 KIVI는 2bit 압축 시 정확도 하락과 긴 컨텍스트에서 메모리 오버헤드가 단점으로 지적된다. InfiniGen은 KV 캐시 크기가 10% 이하로 줄어들 때 정확도 저하가 크고, CPU-GPU 간 전송 병목 문제를 완전히 해소하지 못하는 한계가 있다. 이로 인해 두 기술 모두 압축률과 전송 최적화 간 균형을 맞추는 데 어려움이 존재한다. 판단에 필요한 추가 확인 항목은 긴 컨텍스트 환경에서의 정확도 유지 및 전송 병목 완화 효과의 실증적 검증이다[추론].

두 기술 모두 재학습이나 보정 데이터, 전용 하드웨어 없이 기존 모델에 적용 가능하며, KV 캐시 메모리 문제를 완화하는 접근법을 공유한다. 또한 구현체를 공개하여 개발자 커뮤니티에서 활용되고, 다양한 대형 언어 모델에 적용되어 평가되었다는 점에서 공통점이 존재한다[논문 2402.02750 p.2][논문 2406.19707 p.1-2][웹 https://github.com/jy-yuan/KIVI][웹 https://github.com/snu-comparch/InfiniGen].

## 6. 한계점

1. **공개 정보 기반 추정의 한계** — TRL 4~6 구간은 수율·성능 수치가 비공개라 정보 공백이 가장 크다. 4.1 의 등급은 공개 코드·재현·프레임워크 통합 여부만으로 추정한 것이다.
2. **TRL 기준 시점** — arXiv v1 과 학회 게재·가이드 표기 시점이 논문마다 다르다(KIVI: v1 2024-02 / ICML 2024-07, InfiniGen: v1 2024-06 / OSDI 2024-07). 기준 시점에 따라 추정이 달라진다 — 발표 시점과 채택 간 시차의 구체 사례다.
3. **사전 가설과 확증편향 방지 조치** — 사전 가설 *"온디바이스에서는 SW 압축이 더 적합할 것"* 을 명시하고 장치 8개(기술별 독립 호출 · 사실 단위 질의 · 정확도 임계값 없음 · 근거 없음 기록 · 출처 태그 강제 · 반대 근거 ≥2 · 중립성 검증 루프)를 적용했다. 판정 기준(재학습 불필요 · 전용 HW 불필요)은 배포 용이성에 가중을 두므로 구조적으로 SW 접근에 유리하다 — 도메인의 실제 제약을 반영한 것이지만 기준 선택 자체가 결과에 영향을 준다. Memory · Recall · Latency 3축은 판정이 아닌 해석에만 썼다.
4. **`[추론]` 태그 비율** — 판단 문장 183건 중 논문·웹 근거 없이 추론에만 의존한 문장 5% (9/183).
5. **검색·생성 품질** — Hit Rate@4 0.8 · MRR@4 0.52 (검색 구성 `dual-bm25/ko(dense)+en(sparse)`, 20문항) · RAGAS Faithfulness 0.936 · ResponseRelevancy 0.838 · ContextPrecision 0.912. 검색 24회 중 "논문에 근거 없음" 4건(KIVI 2건 · InfiniGen 2건). 임베딩 비교는 20문항 기준이라 0.10 차이는 2문항이며, 선정은 수치 우위가 아니라 한국어 질의 요건·컨텍스트 길이에 둔다. 근거 없음이 한 기술에 몰리면 그 기술 판정의 `[추론]` 비중이 높아진다. 웹 출처는 Tavily 가 저자·게시일을 주지 않는 경우가 많아 REFERENCE 에 기관명(사이트)과 접근일로 대체했다 — 게시일이 필요한 항목은 사람이 확인해야 한다.
6. **질의 재작성 효과** — 재작성 11건 중 7건이 관련 문서를 찾았다. 효과가 없으면 재작성 단계 제거를 검토한다.

**근거 충분성 미달 셀**
- 근거 부족 셀 market:KIVI (재작업 2/2) — 출처 편중 github.com 88% — 과반
- 근거 부족 셀 market:InfiniGen (재작업 2/2) — 출처 편중 huggingface.co 88% — 과반
- 근거 부족 셀 stakeholder:InfiniGen (재작업 2/2) — 출처 편중 huggingface.co 62% — 과반
- 근거 부족 셀 domain:InfiniGen (재작업 2/2) — 근거 15건 · 반대 2건 · 출처 2곳(최다 93%) · Judge: 근거가 대부분 PC 환경에서의 CPU 메모리 및 PCIe 대역폭 문제에 집중되어 있으며, 스마트폰 온디바이스 환경에서의 실험적 검증이 부족하여 관점에 대한 직접적 답변이 부족함.

## 자동 경고 — 상한 소진으로 종료

- 품질 평가 미달 groundedness (score 0.0) — 판단 문장 태그 20/38(53%) · 추론 단독 7/38(18%) · 본문 미인용 REFERENCE: https://deepwiki.com/snu-comparch/InfiniGen/2-system-architecture, https://deepwiki.com/snu-comparch/InfiniGen/2.1-kv-cache-management, https://huggingface.co/blog/kv-cache-quantization, https://o-mega.ai/articles/google-turboquant-the-2026-llm-compression-guide, https://podcast.do-not-panic.com/2026/06/index.html, https://summerspringwei.github.io/chunweixia.github.io/publication/mobisys26/mobisys26.pdf, https://tianweiz07.github.io/Papers/25-mlsys.pdf, https://www.spheron.network/blog/kv-cache-quantization-with-kivi-int2-setup-guide-2026 · 문제 문장 위치: report 21건, tech_research:KIVI 2건, tech_research:InfiniGen 2건

## REFERENCE

- Liu, Z., Yuan, J., Jin, H. et al.(2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. *Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235*. arXiv:2402.02750.
- Lee, W., Lee, J., Seo, J., Sim, J.(2024). InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*. arXiv:2406.19707.
- jy-yuan(게시일 미상 · 2026-10-07 접근). *GitHub - jy-yuan/KIVI: [ICML 2024] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache · GitHub*. github.com, https://github.com/jy-yuan/KIVI
- alphaxiv.org(게시일 미상 · 2026-10-07 접근). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | alphaXiv*. www.alphaxiv.org, https://www.alphaxiv.org/abs/2402.02750
- spheron.network(게시일 미상 · 2026-10-07 접근). *KV Cache Quantization vLLM Setup: KIVI's INT2 Method (2026) | Spheron Blog*. www.spheron.network, https://www.spheron.network/blog/kv-cache-quantization-with-kivi-int2-setup-guide-2026
- deepwiki.com(게시일 미상 · 2026-10-07 접근). *KV Cache Management*. deepwiki.com, https://deepwiki.com/snu-comparch/InfiniGen/2.1-kv-cache-management
- snu-comparch(게시일 미상 · 2026-10-07 접근). *GitHub - snu-comparch/InfiniGen: InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management (OSDI'24) · GitHub*. github.com, https://github.com/snu-comparch/InfiniGen
- turbo-quant.com(게시일 미상 · 2026-10-07 접근). *TurboQuant vs KIVI: KV Cache 양자화 비교 | TurboQuant Tools*. turbo-quant.com, https://turbo-quant.com/ko/turboquant-vs-kivi
- tianweiz07.github.io(게시일 미상 · 2026-10-07 접근). *Rethinking Key-Value Cache Compression Techniques for ...*. tianweiz07.github.io, https://tianweiz07.github.io/Papers/25-mlsys.pdf
- huggingface.co(게시일 미상 · 2026-10-07 접근). *Unlocking Longer Generation with Key-Value Cache Quantization*. huggingface.co, https://huggingface.co/blog/kv-cache-quantization
- o-mega.ai(게시일 미상 · 2026-10-07 접근). *Google TurboQuant Aug 2026: Deployment & Performance Guide | Articles | o-mega*. o-mega.ai, https://o-mega.ai/articles/google-turboquant-the-2026-llm-compression-guide
- deepwiki.com(게시일 미상 · 2026-10-07 접근). *System Architecture | snu-comparch/InfiniGen | DeepWiki*. deepwiki.com, https://deepwiki.com/snu-comparch/InfiniGen/2-system-architecture
- huggingface.co(게시일 미상 · 2026-10-07 접근). *Paper page - InfiniGen: Efficient Generative Inference of Large Language Models with
  Dynamic KV Cache Management*. huggingface.co, https://huggingface.co/papers/2406.19707
- summerspringwei.github.io(게시일 미상 · 2026-10-07 접근). *KVSwap: Disk-aware KV Cache Offloading for Long- ...*. summerspringwei.github.io, https://summerspringwei.github.io/chunweixia.github.io/publication/mobisys26/mobisys26.pdf
- podcast.do-not-panic.com(게시일 미상 · 2026-10-07 접근). *June 2026 — AI Post Transformers*. podcast.do-not-panic.com, https://podcast.do-not-panic.com/2026/06/index.html
- m.blog.naver.com(게시일 미상 · 2026-10-07 접근). *LPDDR6 특징 분석 온디바이스 AI 시대 핵심 스펙 : 네이버 블로그*. m.blog.naver.com, https://m.blog.naver.com/PostView.naver?blogId=vitamin2230&logNo=224395215402
- ckhome7108.tistory.com(게시일 미상 · 2026-10-07 접근). *온디바이스 AI가 메모리 구조에 미치는 영향*. ckhome7108.tistory.com, https://ckhome7108.tistory.com/entry/%EC%98%A8%EB%94%94%EB%B0%94%EC%9D%B4%EC%8A%A4-AI%EA%B0%80-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EA%B5%AC%EC%A1%B0%EC%97%90-%EB%AF%B8%EC%B9%98%EB%8A%94-%EC%98%81%ED%96%A5
