#!/usr/bin/env python3
"""Build a self-contained Korean function/API glossary for the study package.

The notebooks already contain Korean first-use API notes, but they are difficult
to browse before opening a notebook.  This builder exports the same registry to
an offline HTML reference and adds the package's own NLP/RAG helpers.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectApi:
    key: str
    signature: str
    role: str
    inputs_return: str
    reason: str
    caution: str
    background: str
    project: str


PROJECT_APIS: tuple[ProjectApi, ...] = (
    ProjectApi(
        "normalize_text",
        "normalize_text(text: str) -> str",
        "유니코드 표기를 NFKC로 맞추고, 소문자화와 공백 정리를 한 텍스트로 반환합니다.",
        "문자열 하나를 받아 정규화된 문자열 하나를 돌려줍니다.",
        "같은 뜻의 문장이 서로 다른 공백·대소문자·호환 문자 때문에 다른 토큰으로 취급되는 일을 줄입니다.",
        "원문 표기를 보존해야 하는 화면·법적 기록에는 그대로 쓰면 안 됩니다. 검색·분석용 복사본에 적용하세요.",
        "정규화는 ‘문장의 뜻을 바꾸는 번역’이 아니라, 비교 가능한 형태로 표기를 통일하는 전처리입니다.",
        "01 텍스트 전처리",
    ),
    ProjectApi(
        "mask_pii",
        "mask_pii(text: str) -> str",
        "이메일, 한국 전화번호, 긴 주문번호처럼 개인을 식별할 수 있는 흔한 패턴을 마스킹합니다.",
        "문자열을 받아 일부 위치가 `[EMAIL]`, `[PHONE]` 같은 표식으로 바뀐 문자열을 반환합니다.",
        "로그·학습 데이터·검색 색인에 원문을 넣기 전에 개인정보 노출 가능성을 낮춥니다.",
        "정규표현식 기반의 간단한 보호막입니다. 모든 개인정보를 찾아낸다고 가정하면 안 되고, 실제 서비스는 별도 보안 검토가 필요합니다.",
        "PII는 개인식별정보입니다. 마스킹은 값을 삭제하거나 별도 보관하는 정책과 함께 사용해야 합니다.",
        "01 텍스트 전처리",
    ),
    ProjectApi(
        "tokenize",
        "tokenize(text: str) -> list[str]",
        "정규화된 문장에서 한국어·영어·숫자·마스크 표식을 토큰 목록으로 뽑습니다.",
        "문자열 하나를 받아 순서가 있는 문자열 토큰 목록을 반환합니다.",
        "빈도 계산, TF-IDF, 간단한 분류 모델에 넣을 최소 단위가 필요할 때 사용합니다.",
        "형태소 분석기나 서브워드 토크나이저가 아닙니다. 조사·복합어·띄어쓰기 오류를 완벽히 이해하지 못합니다.",
        "토큰은 모델이 직접 읽는 조각입니다. 어떤 규칙으로 나누는지가 이후 빈도와 검색 결과를 바꿉니다.",
        "01 텍스트 전처리",
    ),
    ProjectApi(
        "corpus_term_frequencies",
        "corpus_term_frequencies(corpus, *, mask_pii_values=True) -> Counter[str]",
        "여러 문장을 정규화·토큰화한 뒤 각 토큰의 출현 횟수를 셉니다.",
        "문자열 iterable을 받아 `토큰 → 횟수` 형태의 Counter를 반환합니다.",
        "어떤 단어가 데이터에서 자주 나타나는지 확인하고, 전처리 결과를 빠르게 검증합니다.",
        "문서 길이가 긴 데이터는 단순 빈도만으로 중요도를 판단하면 과장될 수 있습니다. 검색에는 문서 빈도를 보정하는 TF-IDF가 더 적합합니다.",
        "말뭉치(corpus)는 분석할 문서들의 모음입니다. 빈도는 ‘많이 등장함’이지 ‘의미상 중요함’과 같지 않습니다.",
        "01 텍스트 전처리",
    ),
    ProjectApi(
        "top_terms",
        "top_terms(frequencies, *, limit=20) -> list[tuple[str, int]]",
        "토큰 빈도표에서 횟수가 큰 항목을 정렬해 상위 몇 개만 보여 줍니다.",
        "`토큰 → 횟수` mapping과 개수를 받아 `(토큰, 횟수)` 목록을 반환합니다.",
        "전처리 후 마스크 표식이나 불용어가 지나치게 많이 남았는지 빠르게 눈으로 검사합니다.",
        "동률 처리와 정렬 기준을 확인해야 재현 가능한 결과가 됩니다. 상위 단어만 보고 데이터의 편향을 단정하지 마세요.",
        "빈도표는 데이터 품질 점검 도구입니다. 모델 성능 지표와는 다른 역할입니다.",
        "01 텍스트 전처리",
    ),
    ProjectApi(
        "load_ticket_data",
        "load_ticket_data(path: str | Path) -> pandas.DataFrame",
        "번들된 고객 문의 데이터를 읽고 필수 열·값의 계약을 확인한 DataFrame을 반환합니다.",
        "CSV 경로를 받아 검증된 표 형태 데이터(DataFrame)를 반환합니다.",
        "학습 전에 잘못된 열 이름, 빈 레이블, 형식 오류를 조기에 발견합니다.",
        "파일을 읽었다고 데이터가 학습에 적합하다는 뜻은 아닙니다. 레이블 분포와 누수 여부도 따로 확인하세요.",
        "DataFrame은 행·열로 된 표입니다. 검증은 ‘읽기 성공’과 ‘데이터 계약 만족’을 구분합니다.",
        "02 의도 분류",
    ),
    ProjectApi(
        "TicketClassifierTrainer",
        "TicketClassifierTrainer(config: TrainingConfig)",
        "누수 방지 분할, 파이프라인 학습, 평가와 오류 분석을 한 흐름으로 묶는 분류 학습 도우미입니다.",
        "학습 설정을 받아 내부에서 데이터 분할·모델 적합·평가를 수행하고 TrainingResult를 만듭니다.",
        "분류 실습의 단계를 흩어진 함수 호출이 아니라 한 번의 재현 가능한 작업 단위로 관리합니다.",
        "훈련 데이터와 테스트 데이터를 섞거나, 전처리를 전체 데이터에 먼저 fit하면 성능이 부풀려질 수 있습니다.",
        "데이터 누수는 평가용 정보를 학습에 미리 사용해 실제보다 높은 점수를 만드는 오류입니다.",
        "02 의도 분류",
    ),
    ProjectApi(
        "predict_tickets",
        "predict_tickets(model, tickets) -> pandas.DataFrame",
        "학습된 분류 파이프라인으로 한 문장 또는 여러 문의의 예측 레이블과 점수를 표로 만듭니다.",
        "학습된 Pipeline과 문자열·문자열 목록·DataFrame을 받아 예측 결과 DataFrame을 반환합니다.",
        "학습 정확도만 보지 않고, 실제 입력이 어떤 레이블과 확신도로 분류되는지 확인합니다.",
        "확률이 높아도 정답 보장은 아닙니다. 학습 데이터와 다른 도메인 문장은 별도로 검토해야 합니다.",
        "예측은 결정이고 확률은 모델의 내부 점수입니다. 둘 모두 오류 사례와 함께 해석해야 합니다.",
        "02 의도 분류",
    ),
    ProjectApi(
        "TfidfRetriever",
        "TfidfRetriever(articles: Sequence[KnowledgeArticle])",
        "문자 n-gram TF-IDF 벡터와 cosine similarity로 지식 문서를 찾는 검색기입니다.",
        "지식 문서 목록으로 인덱스를 만들고, `search(question, top_k)`에서 SearchResult 목록을 반환합니다.",
        "생성 모델을 붙이기 전에 검색이 어떤 근거 문서를 가져오는지 투명하게 확인할 수 있습니다.",
        "단어·문자열이 거의 겹치지 않는 동의어에는 약할 수 있습니다. 결과 순위와 출처를 반드시 함께 확인하세요.",
        "TF-IDF는 흔한 단어의 비중을 낮추고 특정 문서에서 특징적인 단어의 비중을 높이는 가중치입니다.",
        "04 RAG from scratch",
    ),
    ProjectApi(
        "load_knowledge_base",
        "load_knowledge_base(path: Path) -> list[KnowledgeArticle]",
        "JSONL 지식 문서를 읽고 스키마와 중복 ID를 검증한 KnowledgeArticle 목록을 만듭니다.",
        "JSONL 파일 경로를 받아 검증된 문서 객체 목록을 반환합니다.",
        "검색 품질을 보기 전에 근거 문서 자체가 올바른 형태인지 보장합니다.",
        "한 줄이 잘못되어도 이후 검색 결과 전체가 흔들릴 수 있습니다. 문서 ID와 제목의 중복을 무시하지 마세요.",
        "JSONL은 한 줄에 JSON 객체 하나를 저장하는 형식입니다. RAG에서는 문서 단위가 인용·평가 단위가 됩니다.",
        "04 RAG from scratch",
    ),
    ProjectApi(
        "format_context",
        "format_context(results, max_chars=2000) -> str",
        "검색 결과를 출처 ID가 남는 제한 길이 컨텍스트 문자열로 합칩니다.",
        "SearchResult iterable과 최대 글자 수를 받아 생성 모델에 넣을 컨텍스트 문자열을 반환합니다.",
        "답변이 어디에서 왔는지 추적할 수 있게 하면서, 프롬프트 길이가 끝없이 커지는 일을 막습니다.",
        "너무 작은 제한은 근거를 자르고, 너무 큰 제한은 잡음·비용·문맥 창 문제를 키웁니다.",
        "컨텍스트는 모델에 전달되는 근거 묶음입니다. 검색 순위와 잘림 규칙이 답변 품질에 영향을 줍니다.",
        "04 RAG from scratch",
    ),
    ProjectApi(
        "build_grounded_prompt",
        "build_grounded_prompt(question, results) -> str",
        "검색 근거 안에서만 답하고, 근거가 없으면 모른다고 말하도록 지시하는 프롬프트를 구성합니다.",
        "질문 문자열과 검색 결과를 받아 근거·질문·답변 규칙이 포함된 문자열을 반환합니다.",
        "RAG의 생성 단계가 검색 결과를 무시하고 그럴듯한 답을 꾸며 내는 위험을 줄입니다.",
        "프롬프트는 보장 장치가 아닙니다. 검색 실패와 모델 환각은 인용·평가로 계속 점검해야 합니다.",
        "Grounded는 답변이 제공된 근거에 묶여 있음을 뜻합니다. 프롬프트와 검색 품질은 함께 작동합니다.",
        "04 RAG from scratch",
    ),
    ProjectApi(
        "DenseSemanticRetriever",
        "DenseSemanticRetriever(articles, n_components=...) ",
        "TF-IDF를 낮은 차원의 밀집 벡터로 줄이고 cosine similarity로 유사한 문서를 찾는 검색기입니다.",
        "문서 목록과 차원 설정으로 인덱스를 만들고, 질의에서 점수가 붙은 SearchResult 목록을 반환합니다.",
        "단어가 완전히 같지 않아도 문서 표현의 방향이 비슷한지를 비교하는 검색 흐름을 실습합니다.",
        "이 구현은 교육용 TF-IDF→SVD 기반입니다. 대규모 임베딩 모델이나 의미 이해를 그대로 대체하지 않습니다.",
        "밀집 임베딩은 고정 길이 숫자 벡터입니다. cosine similarity는 길이보다 벡터 방향의 유사성을 봅니다.",
        "03 의미 검색",
    ),
    ProjectApi(
        "LangChainRAG",
        "LangChainRAG(retriever, generation_chain)",
        "검색 근거를 먼저 노출한 뒤 그 근거를 생성 체인에 전달하는 작은 2단계 RAG 서비스입니다.",
        "검색기와 생성 체인을 받아 `answer(question)`에서 답변과 근거 문서를 함께 반환합니다.",
        "프레임워크를 사용해도 검색·컨텍스트·생성이 어떤 순서로 이어지는지 숨기지 않습니다.",
        "OpenAI 모델 생성은 별도 패키지·API 키가 필요할 수 있습니다. 로컬 해시 임베딩과 실제 모델 결과를 혼동하지 마세요.",
        "RAG 파이프라인은 검색(retrieval)과 생성(generation)을 분리해 근거를 보여 주는 구조입니다.",
        "05 LangChain RAG",
    ),
    ProjectApi(
        "create_openai_model",
        "create_openai_model(model_name: str) -> ChatOpenAI",
        "선택적으로 ChatOpenAI 모델을 만들고, 필요한 추가 패키지가 없을 때 해결 방법을 알려 줍니다.",
        "모델 이름을 받아 LangChain에서 사용할 채팅 모델 객체를 반환합니다.",
        "실습의 로컬 검색 파이프라인에 실제 생성 모델을 연결하고 싶을 때만 사용합니다.",
        "네트워크·API 키·비용이 필요한 선택 기능입니다. 설치가 되지 않았다고 전체 학습이 막히지는 않습니다.",
        "외부 모델 호출은 로컬 코드 실행과 달리 계정·비용·데이터 전송 조건을 동반할 수 있습니다.",
        "05 LangChain RAG",
    ),
    ProjectApi(
        "load_evaluation_cases",
        "load_evaluation_cases(path) -> list[EvaluationCase]",
        "JSONL의 질의·관련 문서 ID·설명으로 구성된 RAG 평가 사례를 읽고 검증합니다.",
        "평가 사례 파일 경로를 받아 EvaluationCase 목록을 반환합니다.",
        "검색기를 바꿨을 때 같은 질의 집합으로 성능을 비교할 수 있게 합니다.",
        "평가 사례가 실제 사용자의 모든 질문을 대표한다고 가정하면 안 됩니다. 누락된 실패 유형을 계속 추가해야 합니다.",
        "평가 세트는 정답을 알고 있는 질문 모음입니다. 학습용 데이터와 분리해야 공정한 비교가 됩니다.",
        "06 RAG 평가",
    ),
    ProjectApi(
        "evaluate_tfidf_retriever",
        "evaluate_tfidf_retriever(retriever, cases, *, top_k) -> RetrievalEvaluationReport",
        "TF-IDF 검색기가 평가 질의마다 관련 문서를 얼마나 상위에 가져오는지 집계합니다.",
        "검색기, 평가 사례, top-k를 받아 recall·MRR·coverage 등이 담긴 보고서를 반환합니다.",
        "‘답변이 그럴듯함’과 별개로 검색 단계 자체의 실패를 수치로 분리해 봅니다.",
        "지표 하나만 높아도 모든 사례가 좋아진 것은 아닙니다. top-k, 사례 수, 실패 사례를 같이 읽으세요.",
        "Recall@k는 관련 문서가 상위 k개 안에 있었는지, MRR은 첫 관련 문서가 얼마나 위에 있었는지를 봅니다.",
        "06 RAG 평가",
    ),
    ProjectApi(
        "get_accelerator",
        "get_accelerator(requested_device=None, mixed_precision=None)",
        "CPU·CUDA·MPS 환경을 확인해 모델과 batch를 옮기고 학습 단계를 도와주는 LabAccelerator를 만듭니다.",
        "원하는 장치·정밀도 설정을 받아 환경에 맞는 가속기 객체를 반환합니다.",
        "같은 노트북이 GPU가 없는 PC에서도 안전하게 실행되도록 장치 선택을 한 곳에 모읍니다.",
        "GPU가 있다고 항상 더 빠르거나 모든 연산이 지원되는 것은 아닙니다. device와 dtype 불일치를 먼저 의심하세요.",
        "device는 연산이 일어나는 CPU/GPU 위치이며, 모델·입력 텐서는 같은 device에 있어야 합니다.",
        "공통 실행 환경",
    ),
)


GLOSSARY_TERMS: tuple[tuple[str, str], ...] = (
    ("tensor", "여러 차원의 숫자 배열입니다. PyTorch에서는 shape, dtype, device, gradient 기록 상태를 함께 가집니다."),
    ("shape", "각 축의 길이를 적은 표기입니다. 예를 들어 `[batch, classes]`는 샘플마다 class 점수 하나씩을 뜻합니다."),
    ("batch", "한 번의 forward/backward에서 함께 처리하는 샘플 묶음입니다."),
    ("gradient", "loss를 조금 줄이기 위해 각 parameter를 어느 방향으로 움직여야 하는지 알려 주는 미분값입니다."),
    ("logit", "softmax를 적용하기 전의 원시 점수입니다. 일반적인 `cross_entropy`에는 logit을 그대로 넣습니다."),
    ("dtype", "값을 저장하는 숫자 형식입니다. 예: float32, int64. 분류 target과 연산 텐서는 요구하는 dtype이 다를 수 있습니다."),
    ("device", "텐서와 모델이 놓인 계산 장치입니다. CPU와 CUDA 텐서는 직접 함께 연산할 수 없습니다."),
    ("embedding", "단어·문서·노드 같은 대상을 고정 길이 숫자 벡터로 표현한 것입니다."),
    ("attention", "query가 key와의 관련도를 점수화하고, 그 비중으로 value를 섞는 연산입니다."),
    ("mask", "연산·loss·attention에서 무시하거나 보호할 위치를 지정하는 불리언/수치 표식입니다."),
    ("retrieval", "질문과 관련된 문서 후보를 찾고 순위를 매기는 단계입니다."),
    ("RAG", "검색 결과를 생성 모델의 근거로 제공하는 Retrieval-Augmented Generation 구조입니다."),
    ("data leakage", "평가에만 있어야 할 정보가 학습 과정에 섞여 실제보다 좋은 성능이 나오는 오류입니다."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the offline Korean function glossary.")
    parser.add_argument("--source-root", type=Path, help="Repository root (defaults to this script's repository).")
    parser.add_argument("--output", type=Path, help="Output HTML path (defaults to <root>/AI_Function_Glossary.html).")
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_api_module(source_root: Path) -> Any:
    """Load the registry without requiring the notebook package at build time."""
    source = source_root / "tools" / "notebook_api_explanations.py"
    if not source.is_file():
        raise FileNotFoundError(f"API registry is missing: {source}")
    # The registry itself is dependency-free.  The module imports nbformat only
    # for functions that edit notebooks, which this exporter never invokes.
    if "nbformat" not in sys.modules:
        sys.modules["nbformat"] = types.ModuleType("nbformat")
    module_name = "_offline_api_registry"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load API registry: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def external_category(key: str, formula: Any | None) -> str:
    lowered = key.lower()
    if formula is not None or any(token in lowered for token in ("attention", "loss", "softmax", "adam", "lstm", "gru", "conv", "linear", "embedding")):
        return "수식 · 모델"
    if lowered.startswith(("torch", "nn.", "f.")) or ".tensor" in lowered:
        return "PyTorch · 텐서"
    if lowered.startswith(("np.", "numpy.", "pd.", "pandas.", "sklearn.")):
        return "데이터 · 수치"
    return "Python · 실행"


def render_card(
    *,
    key: str,
    signature: str,
    role: str,
    inputs_return: str,
    reason: str,
    caution: str,
    category: str,
    aliases: tuple[str, ...] = (),
    docs_url: str | None = None,
    background: str | None = None,
    formula: Any | None = None,
    project: str | None = None,
) -> str:
    search_text = " ".join((key, signature, role, inputs_return, reason, caution, *aliases, category, project or "")).lower()
    docs_link = (
        f'<a class="docs-link" href="{escape(docs_url, quote=True)}" target="_blank" rel="noreferrer">공식 문서 (선택)</a>'
        if docs_url
        else ""
    )
    aliases_html = "".join(f"<code>{escape(alias)}</code>" for alias in aliases if alias != key)
    formula_html = ""
    if formula is not None:
        shape = f'<p><b>shape 연결</b>{escape(formula.shape)}</p>' if getattr(formula, "shape", None) else ""
        formula_html = f'''
          <details class="formula-drawer">
            <summary>수식·shape 연결 보기</summary>
            <div>
              <pre>{escape(formula.equation)}</pre>
              <p><b>기호</b>{escape(formula.symbols)}</p>
              <p><b>코드와 연결</b>{escape(formula.code_bridge)}</p>
              <p><b>직관</b>{escape(formula.intuition)}</p>
              {shape}
            </div>
          </details>'''
    background_html = (
        f'<details class="background-drawer"><summary>필요한 배경 지식</summary><p>{escape(background)}</p></details>'
        if background
        else ""
    )
    project_html = f'<span class="project-tag">{escape(project)}</span>' if project else ""
    return f'''
      <details class="api-card" data-category="{escape(category, quote=True)}" data-search="{escape(search_text, quote=True)}">
        <summary>
          <span class="category-tag">{escape(category)}</span>
          {project_html}
          <code>{escape(key)}</code>
          <span class="card-open">열기</span>
        </summary>
        <div class="api-body">
          <pre class="signature">{escape(signature)}</pre>
          <div class="explanation-grid">
            <section><h3>무엇을 하나요?</h3><p>{escape(role)}</p></section>
            <section><h3>입력과 반환</h3><p>{escape(inputs_return)}</p></section>
            <section><h3>왜 여기서 쓰나요?</h3><p>{escape(reason)}</p></section>
            <section class="caution"><h3>주의할 점</h3><p>{escape(caution)}</p></section>
          </div>
          {background_html}
          {formula_html}
          <div class="api-footer">{docs_link}<span class="aliases">{aliases_html}</span></div>
        </div>
      </details>'''


def render_deep_dive_guide(source_root: Path) -> str:
    """Reuse the long Korean guide as optional reading, not a Markdown file."""
    from build_korean_paper_notes_reader import render_blocks, split_h2_sections

    source = source_root / "docs" / "FUNCTION_API_GUIDE.md"
    if not source.is_file():
        raise FileNotFoundError(f"Function API guide is missing: {source}")
    lines = source.read_text(encoding="utf-8").splitlines()
    intro, sections = split_h2_sections(lines)
    intro = [line for line in intro if not line.startswith("# ")]
    intro_html = render_blocks(intro, source.parent, source_root)
    topic_html = "".join(
        f'<details class="deep-topic"><summary>{escape(heading)}</summary>'
        f'<div>{render_blocks(body, source.parent, source_root)}</div></details>'
        for heading, body in sections
    )
    return f'''
      <details class="deep-guide">
        <summary>함수·수식 심화 가이드 25개 열기</summary>
        <div class="deep-guide-body">
          <p>카드의 짧은 설명으로 부족할 때 읽는 긴 한국어 해설입니다. 비교, 난수, 학습 루프, shape, device, loss, attention, AdamW까지 단계적으로 다룹니다.</p>
          <div class="deep-intro">{intro_html}</div>
          {topic_html}
        </div>
      </details>'''


def build_html(source_root: Path) -> str:
    registry_module = load_api_module(source_root)
    registry = registry_module.API_REGISTRY
    formulas = registry_module.API_FORMULAS
    cards: list[str] = []
    categories: set[str] = set()

    for key, entry in sorted(registry.items(), key=lambda item: item[0].lower()):
        formula = formulas.get(key)
        category = external_category(key, formula)
        categories.add(category)
        cards.append(
            render_card(
                key=key,
                signature=entry.signature,
                role=entry.role,
                inputs_return=entry.inputs_return,
                reason=entry.reason,
                caution=entry.caution,
                category=category,
                aliases=entry.aliases,
                docs_url=entry.docs_url,
                formula=formula,
            )
        )

    for entry in PROJECT_APIS:
        category = "프로젝트 전용 함수"
        categories.add(category)
        cards.append(
            render_card(
                key=entry.key,
                signature=entry.signature,
                role=entry.role,
                inputs_return=entry.inputs_return,
                reason=entry.reason,
                caution=entry.caution,
                category=category,
                background=entry.background,
                project=entry.project,
            )
        )

    filter_buttons = '<button type="button" class="filter active" data-filter="all">전체</button>' + "".join(
        f'<button type="button" class="filter" data-filter="{escape(category, quote=True)}">{escape(category)}</button>'
        for category in sorted(categories)
    )
    glossary = "".join(f"<dt>{escape(term)}</dt><dd>{escape(definition)}</dd>" for term, definition in GLOSSARY_TERMS)
    deep_dive = render_deep_dive_guide(source_root)
    total_cards = len(cards)
    return f'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI 함수·API 한국어 사전</title>
  <style>
    :root {{ --ink:#172033; --muted:#5e6b7f; --line:#dbe3ee; --page:#f4f7fb; --navy:#152b4d; --blue:#2f6feb; --blue-soft:#eaf2ff; --green:#0f766e; --amber:#a16207; --reading-size:16px; }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }} body {{ margin:0; color:var(--ink); background:var(--page); font:var(--reading-size)/1.74 "Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif; }}
    button,input {{ font:inherit; }} button {{ cursor:pointer; }} a {{ color:#1558c0; text-underline-offset:3px; }}
    .shell {{ max-width:1230px; margin:0 auto; padding:26px 24px 72px; }}
    .hero {{ padding:39px 44px; border-radius:23px; color:white; background:radial-gradient(circle at 85% 15%,rgba(76,170,255,.38),transparent 26%),linear-gradient(130deg,#142747,#25558c); box-shadow:0 14px 34px rgba(21,39,70,.13); }}
    .eyebrow {{ margin:0 0 10px; color:#8cbcff; font-size:13px; font-weight:800; letter-spacing:.09em; text-transform:uppercase; }}
    h1 {{ max-width:760px; margin:0; font-size:clamp(30px,5vw,46px); line-height:1.22; letter-spacing:-.045em; }}
    .hero p:not(.eyebrow) {{ max-width:850px; margin:15px 0 0; color:#dce9ff; }}
    .stats {{ display:flex; flex-wrap:wrap; gap:11px; margin-top:23px; }} .stat {{ min-width:150px; padding:10px 14px; border:1px solid rgba(255,255,255,.23); border-radius:12px; background:rgba(255,255,255,.10); }} .stat b {{ display:block; font-size:21px; }} .stat span {{ color:#dce9ff; font-size:13px; }}
    .toolbar {{ position:sticky; top:0; z-index:4; display:flex; gap:10px; align-items:center; padding:14px 0; background:linear-gradient(var(--page) 78%,rgba(244,247,251,0)); }}
    .search {{ position:relative; flex:1; }} .search:before {{ content:"⌕"; position:absolute; top:7px; left:14px; color:#526982; font-size:25px; line-height:1; }} .search input {{ width:100%; padding:12px 14px 12px 41px; border:1px solid #cad7e7; border-radius:12px; color:var(--ink); background:white; }}
    .tool {{ min-height:43px; padding:8px 12px; border:1px solid #cad7e7; border-radius:10px; color:#29496d; background:white; font-size:14px; font-weight:700; }} .tool:hover {{ background:#f1f6ff; }}
    .intro, .results {{ padding:28px; border:1px solid var(--line); border-radius:20px; background:white; box-shadow:0 6px 21px rgba(24,48,85,.06); }} .intro h2 {{ margin:0; font-size:26px; letter-spacing:-.03em; }} .intro>p {{ margin:9px 0 20px; color:var(--muted); }}
    .principles {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }} .principle {{ padding:17px; border-radius:13px; background:#f3f7fd; }} .principle b {{ display:block; margin-bottom:5px; color:#204b86; }} .principle span {{ color:#536177; font-size:14px; line-height:1.6; }}
    .term-drawer,.deep-guide {{ margin-top:20px; overflow:clip; border:1px solid #c8ddd8; border-radius:12px; background:#f5fcfa; }} .term-drawer summary,.deep-guide>summary {{ padding:13px 16px; color:#075f57; cursor:pointer; font-weight:800; }} .term-drawer>div {{ padding:0 17px 17px; }} dl {{ display:grid; grid-template-columns:150px 1fr; gap:0; margin:0; border-top:1px solid #d7ebe5; }} dt,dd {{ margin:0; padding:9px 0; border-bottom:1px solid #d7ebe5; }} dt {{ color:#0c7065; font-weight:800; }} dd {{ color:#485b67; }}
    .deep-guide {{ border-color:#cfdcf0; background:#f7faff; }} .deep-guide>summary {{ color:#24528c; }} .deep-guide-body {{ padding:0 17px 17px; color:#43546b; }} .deep-guide-body>p {{ margin-top:0; }} .deep-intro {{ padding:14px; border-radius:9px; background:white; }} .deep-topic {{ margin-top:10px; overflow:clip; border:1px solid #dce6f3; border-radius:9px; background:white; }} .deep-topic summary {{ padding:11px 13px; color:#244e82; cursor:pointer; font-size:14px; font-weight:800; }} .deep-topic>div {{ padding:0 14px 14px; }} .deep-topic h3 {{ color:#315c96; font-size:15px; }} .deep-topic pre {{ overflow:auto; padding:12px; border-radius:8px; color:#dce9ff; background:#192a43; font-size:12px; white-space:pre-wrap; }} .deep-topic code {{ padding:1px 4px; border-radius:4px; color:#6b4619; background:#fff4e6; font-size:.9em; }} .deep-topic .display-math {{ overflow:auto; padding:11px; border-left:3px solid #80a8e8; border-radius:8px; color:#334982; background:#edf4ff; font-family:Cambria Math,Cambria,serif; }} .deep-topic .source-note {{ padding:11px; border-radius:8px; color:#695622; background:#fff9e7; }}
    .results {{ margin-top:17px; }} .filter-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:13px; }} .filter {{ padding:7px 10px; border:1px solid #cbd9eb; border-radius:999px; color:#355474; background:white; font-size:13px; font-weight:700; }} .filter.active,.filter:hover {{ border-color:#6f9feb; color:white; background:#2f6feb; }} .result-count {{ margin:0; color:var(--muted); font-size:14px; }}
    .api-list {{ display:grid; gap:11px; margin-top:14px; }} .api-card {{ overflow:clip; border:1px solid #d9e4f0; border-radius:13px; background:white; }} .api-card[open] {{ border-color:#aac6ef; box-shadow:0 7px 18px rgba(30,81,151,.08); }} .api-card summary {{ display:flex; gap:9px; align-items:center; padding:15px 17px; list-style:none; cursor:pointer; }} .api-card summary::-webkit-details-marker {{ display:none; }} .api-card summary:before {{ content:"+"; color:#2f6feb; font-size:20px; font-weight:800; }} .api-card[open] summary:before {{ content:"−"; }} .api-card summary>code {{ min-width:0; color:#1d2c42; font:700 15px/1.5 Consolas,"Cascadia Code",monospace; overflow-wrap:anywhere; }} .category-tag,.project-tag {{ flex:0 0 auto; padding:2px 7px; border-radius:999px; color:#1d5cae; background:#eaf2ff; font-size:11px; font-weight:800; }} .project-tag {{ color:#0a6e62; background:#e6f7f2; }} .card-open {{ margin-left:auto; color:#728198; font-size:12px; font-weight:600; }}
    .api-body {{ padding:0 18px 20px 47px; }} pre.signature {{ overflow:auto; margin:0 0 14px; padding:12px 14px; border-radius:9px; color:#e6efff; background:#192a43; font:13px/1.6 Consolas,"Cascadia Code",monospace; white-space:pre-wrap; }} .explanation-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }} .explanation-grid section {{ padding:13px; border:1px solid #e2e9f1; border-radius:10px; background:#fbfdff; }} .explanation-grid .caution {{ border-color:#f2dfb6; background:#fffaf0; }} .explanation-grid h3 {{ margin:0 0 6px; color:#22518f; font-size:14px; }} .explanation-grid .caution h3 {{ color:#875a0a; }} .explanation-grid p {{ margin:0; color:#3e4e62; font-size:14px; }}
    .background-drawer,.formula-drawer {{ margin-top:11px; border:1px solid #d6e4ef; border-radius:9px; background:#f8fbff; }} .background-drawer summary,.formula-drawer summary {{ padding:10px 12px; color:#31597e; cursor:pointer; font-size:14px; font-weight:800; }} .background-drawer p,.formula-drawer>div {{ margin:0; padding:0 13px 13px; color:#44566c; font-size:14px; }} .formula-drawer pre {{ overflow:auto; margin:0 0 10px; padding:11px; border-radius:7px; color:#344680; background:#edf3ff; white-space:pre-wrap; }} .formula-drawer p {{ margin:8px 0; }} .formula-drawer b {{ display:block; margin-bottom:2px; color:#254f86; }}
    .api-footer {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:10px; align-items:center; margin-top:13px; }} .docs-link {{ font-size:13px; font-weight:800; }} .aliases {{ display:flex; flex-wrap:wrap; gap:5px; }} .aliases code {{ padding:2px 5px; border-radius:4px; color:#63431d; background:#fff4e6; font-size:12px; }} [hidden] {{ display:none!important; }}
    @media (max-width:760px) {{ .shell {{ padding:13px 13px 42px; }} .hero {{ padding:29px 22px; border-radius:18px; }} .toolbar {{ flex-wrap:wrap; }} .search {{ flex-basis:100%; }} .intro,.results {{ padding:21px 16px; border-radius:16px; }} .principles,.explanation-grid {{ grid-template-columns:1fr; }} dl {{ grid-template-columns:1fr; }} dt {{ padding-bottom:2px; border-bottom:0; }} dd {{ padding-top:2px; }} .api-body {{ padding:0 13px 16px; }} .api-card summary {{ padding:13px; gap:7px; }} .category-tag,.project-tag {{ display:none; }} .card-open {{ display:none; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">offline reference · 한국어 함수 사전</p>
      <h1>인터넷 검색 없이<br>코드의 함수를 이해하는 화면</h1>
      <p>이 과정에서 실제로 쓰는 라이브러리 API와 프로젝트 전용 함수를 한국어로 설명합니다. 카드마다 무엇을 입력하고 무엇을 반환하는지, 왜 이 위치에서 쓰는지, 무엇을 조심해야 하는지가 들어 있습니다.</p>
      <div class="stats"><div class="stat"><b>{len(registry)}개</b><span>노트북 API 설명</span></div><div class="stat"><b>{len(PROJECT_APIS)}개</b><span>NLP·RAG 프로젝트 함수</span></div><div class="stat"><b>{len(formulas)}개</b><span>수식·shape 연결</span></div></div>
    </header>
    <div class="toolbar" aria-label="함수 사전 도구">
      <label class="search"><input id="api-search" type="search" placeholder="함수명 또는 한국어 설명으로 찾기 (예: cross entropy, 텐서, RAG)" autocomplete="off"></label>
      <button id="font-down" class="tool" type="button">가−</button><button id="font-up" class="tool" type="button">가＋</button>
    </div>
    <section id="overview" class="intro">
      <h2>카드를 이렇게 읽으면 됩니다</h2>
      <p>코드를 외우기보다, 함수가 받는 값·돌려주는 값·이 위치에서의 역할·shape/device/부작용을 먼저 확인하세요. 공식 문서는 선택 링크일 뿐, 핵심 설명은 이 화면 안에 있습니다.</p>
      <div class="principles"><div class="principle"><b>입력과 반환</b><span>무엇을 넣고 어떤 타입·shape가 나오는지 먼저 봅니다.</span></div><div class="principle"><b>왜 이 함수인가</b><span>동일한 일을 할 수 있는 여러 API 중 이 코드에서 필요한 이유를 확인합니다.</span></div><div class="principle"><b>실수 방지</b><span>dtype, device, 상태 변경, 데이터 누수처럼 결과를 바꾸는 함정을 확인합니다.</span></div></div>
      <details class="term-drawer"><summary>배경 용어 사전 열기</summary><div><dl>{glossary}</dl></div></details>
      {deep_dive}
    </section>
    <section class="results">
      <div class="filter-row">{filter_buttons}</div>
      <p id="result-count" class="result-count">{total_cards}개 항목을 표시합니다.</p>
      <div class="api-list">{''.join(cards)}</div>
    </section>
  </main>
  <script>
    (() => {{
      const cards = [...document.querySelectorAll('.api-card')];
      const search = document.getElementById('api-search');
      const count = document.getElementById('result-count');
      const buttons = [...document.querySelectorAll('.filter')];
      const root = document.documentElement;
      let filter = 'all'; let fontSize = 16;
      function update() {{
        const query = search.value.trim().toLocaleLowerCase(); let visible = 0;
        cards.forEach(card => {{
          const textMatch = !query || card.dataset.search.includes(query);
          const categoryMatch = filter === 'all' || card.dataset.category === filter;
          const show = textMatch && categoryMatch;
          card.hidden = !show; if (show) visible += 1;
        }});
        count.textContent = query || filter !== 'all' ? `검색 결과 ${{visible}}개` : `${{visible}}개 항목을 표시합니다.`;
      }}
      search.addEventListener('input', update);
      buttons.forEach(button => button.addEventListener('click', () => {{ filter = button.dataset.filter; buttons.forEach(other => other.classList.toggle('active', other === button)); update(); }}));
      document.getElementById('font-down').addEventListener('click', () => {{ fontSize = Math.max(14, fontSize - 1); root.style.setProperty('--reading-size', `${{fontSize}}px`); }});
      document.getElementById('font-up').addEventListener('click', () => {{ fontSize = Math.min(23, fontSize + 1); root.style.setProperty('--reading-size', `${{fontSize}}px`); }});
    }})();
  </script>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    source_root = (args.source_root or repository_root()).resolve()
    output = (args.output or source_root / "AI_Function_Glossary.html").resolve()
    html_text = build_html(source_root)
    html_text = "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(f"WROTE {output}")
    print("PROJECT_API_CARDS=" + str(len(PROJECT_APIS)))


if __name__ == "__main__":
    main()
