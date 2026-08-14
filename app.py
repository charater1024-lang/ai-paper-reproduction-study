"""실시간으로 데이터·모델·검색 동작을 관찰하는 Streamlit 실습 앱."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import pandas as pd
    import streamlit as st
except ImportError as error:  # 직접 실행했을 때 친절한 설치 안내
    raise SystemExit(
        '앱 의존성이 없습니다. `pip install -e ".[all]"` 후 실행하세요.'
    ) from error

from llm_engineering_lab.retrieval import (  # noqa: E402
    TfidfRetriever,
    build_grounded_prompt,
    load_knowledge_base,
)

st.set_page_config(page_title="LLM Engineering Lab", page_icon="🧪", layout="wide")


@st.cache_data
def load_ticket_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def get_retriever(path: str) -> TfidfRetriever:
    return TfidfRetriever().fit(load_knowledge_base(Path(path)))


@st.cache_resource
def load_torch_classifier(path: str, modified_ns: int):
    """mtime을 cache key에 포함해 새 checkpoint가 생기면 자동으로 다시 읽는다."""

    del modified_ns
    from llm_engineering_lab.acceleration import get_accelerator
    from llm_engineering_lab.torch_text import load_classifier_checkpoint

    model, payload = load_classifier_checkpoint(Path(path), map_location="cpu")
    accelerator = get_accelerator()
    return accelerator.move(model).eval(), payload, accelerator


@st.cache_resource
def load_tiny_lm(path: str, modified_ns: int):
    del modified_ns
    from llm_engineering_lab.acceleration import get_accelerator
    from llm_engineering_lab.transformer import load_tiny_lm_checkpoint

    model, payload = load_tiny_lm_checkpoint(Path(path), map_location="cpu")
    accelerator = get_accelerator()
    return accelerator.move(model).eval(), payload, accelerator


def render_overview() -> None:
    st.subheader("실습 지도")
    st.markdown(
        """
        이 앱은 완성품보다 **관찰 도구**에 가깝습니다. 터미널에서 각 학습 스크립트를
        실행한 뒤 여기서 데이터 분포, 예측 확률, 검색 점수, 생성 설정의 영향을 확인하세요.

        1. `01_inspect_data.py` — 스키마, 중복, 누수 후보 관찰
        2. `02_train_ml_baseline.py` — TF-IDF 기준 모델과 오류 분석
        3. `03_train_torch_classifier.py` — 배치/손실/gradient 기반 학습
        4. `04_train_tiny_lm.py` — causal mask와 next-token prediction
        5. `05_llm_inference_patterns.py` — batching·sampling·안전한 요청 패턴
        6. `06_rag_demo.py` — retrieval, recall@k, grounded prompt
        """
    )
    st.info(
        "모든 예제는 작은 로컬 데이터로 동작합니다. 실제 고객 정보나 API 키를 넣지 마세요."
    )


def render_data_lab() -> None:
    st.subheader("고객지원 티켓 데이터")
    csv_path = ROOT / "data" / "customer_support_tickets.csv"
    if not csv_path.exists():
        st.warning(
            "데이터 파일이 아직 없습니다. README의 데이터 생성/검사 단계를 실행하세요."
        )
        return
    frame = load_ticket_frame(str(csv_path))
    labels = sorted(frame["label"].dropna().unique())
    chosen = st.multiselect("라벨 필터", labels, default=labels)
    filtered = frame[frame["label"].isin(chosen)]

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 행", f"{len(frame):,}")
    col2.metric("필터 행", f"{len(filtered):,}")
    col3.metric("라벨 수", frame["label"].nunique())
    counts = filtered["label"].value_counts().rename_axis("label").to_frame("count")
    st.bar_chart(counts)
    st.dataframe(filtered.head(100), width="stretch", hide_index=True)

    if "case_id" in frame.columns:
        duplicate_cases = int(frame["case_id"].duplicated(keep=False).sum())
        st.caption(
            f"같은 case_id를 공유하는 행: {duplicate_cases}개. "
            "랜덤 행 분할이 왜 데이터 누수를 만들 수 있는지 확인해 보세요."
        )


def _find_baseline_model() -> Path | None:
    candidates = (
        ROOT / "artifacts" / "ml" / "ticket_classifier.joblib",
        ROOT / "artifacts" / "ml_baseline.joblib",
        ROOT / "artifacts" / "baseline_model.joblib",
        ROOT / "artifacts" / "ticket_classifier.joblib",
    )
    return next((path for path in candidates if path.exists()), None)


def render_tensor_lab() -> None:
    st.subheader("PyTorch 토큰·텐서 해부")
    checkpoint = ROOT / "artifacts" / "ticket_classifier.pt"
    if not checkpoint.exists():
        st.info(
            "먼저 `python scripts/03_train_torch_classifier.py --device auto`를 실행하세요."
        )
        return
    try:
        import torch

        from llm_engineering_lab.torch_text import Vocabulary
    except ImportError:
        st.warning("PyTorch가 없습니다. `pip install -e .`로 의존성을 설치하세요.")
        return

    model, payload, accelerator = load_torch_classifier(
        str(checkpoint), checkpoint.stat().st_mtime_ns
    )
    st.caption(accelerator.summary())
    extra = payload["extra"]
    vocabulary = Vocabulary.from_dict(extra["vocabulary"])
    label_to_id = {
        str(label): int(index) for label, index in extra["label_to_id"].items()
    }
    id_to_label = {index: label for label, index in label_to_id.items()}
    max_length = int(extra.get("max_length", 64))

    raw = st.text_area(
        "한 줄에 하나씩 입력",
        "카드가 두 번 결제됐어요\nAPI 요청에서 429 오류가 납니다",
        height=110,
    )
    messages = [line.strip() for line in raw.splitlines() if line.strip()]
    if not messages:
        st.warning("한 개 이상의 문장을 입력하세요.")
        return

    encoded = [
        vocabulary.encode(text, add_bos=True, add_eos=True, max_length=max_length)
        for text in messages
    ]
    width = max(map(len, encoded))
    input_ids = torch.full((len(encoded), width), vocabulary.pad_id, dtype=torch.long)
    for row, token_ids in enumerate(encoded):
        input_ids[row, : len(token_ids)] = torch.tensor(token_ids)
    attention_mask = input_ids.ne(vocabulary.pad_id)
    input_ids, attention_mask = accelerator.move(input_ids, attention_mask)

    with torch.inference_mode():
        logits = model(input_ids, attention_mask)
        probabilities = logits.softmax(dim=-1)

    shape1, shape2, shape3 = st.columns(3)
    shape1.metric("input_ids", str(tuple(input_ids.shape)))
    shape2.metric("attention_mask", str(tuple(attention_mask.shape)))
    shape3.metric("logits", str(tuple(logits.shape)))
    st.caption("B=batch, T=현재 batch의 최대 token 길이, C=의도 클래스 수")

    first_ids = input_ids[0].cpu().tolist()
    token_table = pd.DataFrame(
        {
            "position": range(len(first_ids)),
            "token": [
                vocabulary.decode([token_id], skip_special_tokens=False)
                for token_id in first_ids
            ],
            "token_id": first_ids,
            "attention": attention_mask[0].cpu().tolist(),
        }
    )
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### 첫 문장의 token sequence")
        st.dataframe(token_table, width="stretch", hide_index=True)
    with right:
        st.markdown("#### 실제 tensor")
        st.code(
            f"input_ids =\n{input_ids}\n\nattention_mask =\n{attention_mask}",
            language="text",
        )

    prediction_rows = []
    for text, scores in zip(messages, probabilities.cpu(), strict=True):
        predicted_id = int(scores.argmax())
        prediction_rows.append(
            {
                "text": text,
                "prediction": id_to_label[predicted_id],
                "confidence": float(scores[predicted_id]),
            }
        )
    st.markdown("#### forward 결과")
    st.dataframe(pd.DataFrame(prediction_rows), width="stretch", hide_index=True)
    st.caption(
        "문장 수와 길이를 바꾸며 B와 T가 어떻게 변하고 C는 왜 고정되는지 관찰하세요."
    )


def render_classifier_lab() -> None:
    st.subheader("분류 모델 추론")
    model_path = _find_baseline_model()
    if model_path is None:
        st.info(
            "먼저 `python scripts/02_train_ml_baseline.py`를 실행해 모델을 학습하세요."
        )
        return

    text = st.text_area(
        "새 티켓",
        "어제부터 로그인이 안 되고 비밀번호 재설정 메일도 오지 않아요",
        height=100,
    )
    if st.button("의도 분류", type="primary"):
        try:
            from llm_engineering_lab.ml import load_model, predict_tickets

            model = load_model(model_path)
            prediction = predict_tickets(model, [text]).iloc[0]
            st.success(
                f"예측 결과: {prediction['predicted_label']} "
                f"(confidence={prediction['confidence']:.3f})"
            )
            probability_columns = [
                column for column in prediction.index if column.startswith("prob_")
            ]
            chart = pd.DataFrame(
                {
                    "label": [
                        column.removeprefix("prob_") for column in probability_columns
                    ],
                    "probability": [
                        float(prediction[column]) for column in probability_columns
                    ],
                }
            ).sort_values("probability", ascending=False)
            st.bar_chart(chart.set_index("label"))
            st.caption(
                "확률이 비슷한 클래스를 모아 오류 분석용 평가셋을 만들어 보세요."
            )
        except (
            Exception
        ) as error:  # 앱에서는 실패 맥락을 보여 주는 것이 학습에 유용하다.
            st.exception(error)


def render_rag_lab() -> None:
    st.subheader("검색 증강 생성(RAG) 해부")
    kb_path = ROOT / "data" / "raw" / "knowledge_base.jsonl"
    retriever = get_retriever(str(kb_path))
    query = st.text_input("사용자 질문", "API 요청에서 429 오류가 계속 발생해요")
    top_k = st.slider("top-k", min_value=1, max_value=6, value=3)
    min_score = st.slider("표시할 최소 유사도", 0.0, 0.5, 0.01, 0.01)

    started = time.perf_counter()
    results = [
        result for result in retriever.search(query, top_k) if result.score >= min_score
    ]
    latency_ms = (time.perf_counter() - started) * 1_000
    st.caption(f"검색 지연 {latency_ms:.2f} ms · {len(results)}개 문서")

    if not results:
        st.warning(
            "임계값을 넘은 근거가 없습니다. 이때 답변을 거부하거나 질문을 되묻는 정책이 필요합니다."
        )
    for result in results:
        with st.expander(
            f"#{result.rank} {result.article.title} · score {result.score:.3f}",
            expanded=result.rank == 1,
        ):
            st.write(result.article.content)
            st.caption(
                f"source={result.article.id} · category={result.article.category}"
            )

    st.markdown("#### LLM에 전달할 프롬프트")
    st.code(build_grounded_prompt(query, results), language="text")
    st.caption(
        "질문이나 검색 문서가 프롬프트의 지시문을 덮어쓰지 못하도록 방어 규칙을 추가해 보세요."
    )


def render_generation_lab() -> None:
    st.subheader("Tiny LM 생성 실험")
    checkpoint = ROOT / "artifacts" / "tiny_lm.pt"
    if not checkpoint.exists():
        st.info("먼저 `python scripts/04_train_tiny_lm.py --device auto`를 실행하세요.")
        return
    try:
        import torch

        from llm_engineering_lab.torch_text import Vocabulary
    except ImportError:
        st.warning("PyTorch가 없습니다. `pip install -e .`로 의존성을 설치하세요.")
        return

    model, payload, accelerator = load_tiny_lm(
        str(checkpoint), checkpoint.stat().st_mtime_ns
    )
    st.caption(accelerator.summary())
    vocabulary = Vocabulary.from_dict(payload["extra"]["vocabulary"])
    prompt = st.text_input("prompt", "고객의 결제가 두 번 처리됐어요")
    col1, col2, col3 = st.columns(3)
    temperature = col1.slider("temperature", 0.0, 1.5, 0.8, 0.1)
    top_k = col2.slider("top-k (0=끄기)", 0, min(50, len(vocabulary)), 20)
    max_new_tokens = col3.slider("새 token 수", 1, 24, 12)
    use_cache = st.toggle("KV cache 사용", value=True)
    seed = st.number_input("seed", min_value=0, max_value=100_000, value=42)

    max_prompt_length = model.config.max_seq_length - max_new_tokens
    prompt_ids = vocabulary.encode(prompt, add_bos=True, max_length=max_prompt_length)
    st.caption(
        f"prompt tokens={len(prompt_ids)} · context limit={model.config.max_seq_length} · "
        f"checkpoint epoch={payload.get('epoch', '?')}"
    )

    if st.button("token 생성", type="primary"):
        if accelerator.device.type in {"cpu", "cuda"}:
            generator = torch.Generator(device=accelerator.device.type).manual_seed(
                int(seed)
            )
        else:
            torch.manual_seed(int(seed))
            generator = None
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=accelerator.device
        )
        accelerator.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=None if top_k == 0 else top_k,
                top_p=0.95,
                eos_id=vocabulary.eos_id,
                use_cache=use_cache,
                generator=generator,
            )
        accelerator.synchronize()
        latency_ms = (time.perf_counter() - started) * 1_000
        generated_ids = generated[0].cpu().tolist()
        completion_ids = generated_ids[len(prompt_ids) :]
        st.success(vocabulary.decode(generated_ids))
        st.write(
            {
                "latency_ms": round(latency_ms, 2),
                "generated_tokens": len(completion_ids),
                "completion_token_ids": completion_ids,
                "kv_cache": use_cache,
            }
        )
        st.caption(
            "같은 seed를 고정한 뒤 temperature, top-k, KV cache를 하나씩 바꿔 비교하세요."
        )


def render_artifacts() -> None:
    st.subheader("실험 산출물")
    artifacts = ROOT / "artifacts"
    files = [
        path
        for path in artifacts.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    if not files:
        st.info(
            "학습 스크립트를 실행하면 체크포인트와 메트릭 파일이 여기에 나타납니다."
        )
        return
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        size_kb = path.stat().st_size / 1024
        with st.expander(f"{relative} · {size_kb:.1f} KB"):
            if path.suffix == ".json":
                try:
                    st.json(json.loads(path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    st.write("JSON 미리보기에 실패했습니다.")
            else:
                st.write("바이너리 또는 체크포인트 파일입니다.")


def main() -> None:
    st.title("🧪 LLM Engineering Lab")
    st.caption(
        "Python · 전통 ML · PyTorch · Tiny Transformer · RAG를 한 흐름으로 연습합니다."
    )
    page = st.sidebar.radio(
        "실험실 선택",
        (
            "시작",
            "데이터",
            "PyTorch 해부",
            "분류 추론",
            "Tiny LM 생성",
            "RAG 검색",
            "산출물",
        ),
    )
    renderers = {
        "시작": render_overview,
        "데이터": render_data_lab,
        "PyTorch 해부": render_tensor_lab,
        "분류 추론": render_classifier_lab,
        "Tiny LM 생성": render_generation_lab,
        "RAG 검색": render_rag_lab,
        "산출물": render_artifacts,
    }
    renderers[page]()


if __name__ == "__main__":
    main()
