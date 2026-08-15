#!/usr/bin/env python3
"""Build a standalone, browser-friendly Korean AI paper-notes reader.

The portable package intentionally keeps its source notes as Markdown.  That is
excellent for maintenance, but a Windows association can send a learner to VS
Code instead of a comfortable reading view.  This small dependency-free builder
embeds the existing Korean notes in one offline HTML reader, so it works from a
file:// browser tab and never needs a local web server.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import quote, urlparse


@dataclass(frozen=True)
class FieldSpec:
    slug: str
    filename: str
    label: str
    short_label: str
    route: str


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("vision", "VISION.md", "컴퓨터 비전", "LeNet-5 → DETR", "이미지·검출·Transformer"),
    FieldSpec("nlp_llm", "NLP_LLM.md", "NLP · LLM", "word2vec → RAG", "언어·검색·생성"),
    FieldSpec("generative", "GENERATIVE.md", "생성 모델", "DAE → Latent Diffusion", "확률·GAN·확산"),
    FieldSpec("reinforcement_learning", "REINFORCEMENT_LEARNING.md", "강화학습 · 에이전트", "REINFORCE → AlphaZero", "보상·정책·탐색"),
    FieldSpec("graph_recommendation", "GRAPH_RECOMMENDATION.md", "그래프 · 추천", "DeepWalk → LightGCN", "그래프·표현·랭킹"),
    FieldSpec("self_supervised_multimodal", "SELF_SUPERVISED_MULTIMODAL.md", "자기지도 · 멀티모달", "CPC → Flamingo", "대조학습·비전언어"),
    FieldSpec("distillation_compression", "DISTILLATION_COMPRESSION.md", "지식 증류 · 경량화", "KD → Once-for-All", "압축·배포·효율"),
)

DOC_ANCHORS = {
    "README.md": "#overview",
    "AI_ASSISTED_NOTICE.md": "#notice",
    "RECENT_TOP_TIER.md": "#radar",
    **{field.filename: f"#field-{field.slug}" for field in FIELDS},
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^(\s*)(?:([-*+])|(\d+)\.)\s+(.+?)\s*$")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
NUMBERED_PAPER_RE = re.compile(r"^(\d{2})\.\s+(.+)$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:---+|\*\*\*+)\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
STRONG_RE = re.compile(r"\*\*(.+?)\*\*")
CODE_RE = re.compile(r"`([^`]+)`")
MATH_RE = re.compile(r"\$([^$\n]+)\$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the offline Korean paper-notes reader.")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Repository root that contains docs/paper_reading_notes (defaults to this script's repository).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (defaults to <source-root>/AI_Korean_Paper_Notes.html).",
    )
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_external_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https", "mailto"}


def safe_href(raw_href: str, source_dir: Path, source_root: Path) -> str:
    """Make local Markdown links work from a reader placed at repository root."""
    href = raw_href.strip()
    if not href or href.startswith("#") or is_external_url(href):
        return href or "#"

    # A few source links have a title after the URL.  The notes do not depend on
    # it, so keeping the target itself is more useful than carrying the title.
    path_part = href.split(maxsplit=1)[0]
    file_part, separator, fragment = path_part.partition("#")
    candidate = (source_dir / file_part).resolve()

    if candidate.suffix.lower() == ".md":
        anchor = DOC_ANCHORS.get(candidate.name)
        if anchor:
            return anchor

    try:
        relative = candidate.relative_to(source_root).as_posix()
    except ValueError:
        # Leave an unusual path untouched rather than creating an invalid link.
        return href

    encoded = quote(relative, safe="/-_.~")
    return f"{encoded}#{fragment}" if separator and fragment else encoded


def inline_html(text: str, source_dir: Path, source_root: Path) -> str:
    """Render the small, well-defined inline Markdown subset used by the notes."""
    links: list[str] = []

    def protect_link(match: re.Match[str]) -> str:
        label, raw_href = match.groups()
        href = safe_href(raw_href, source_dir, source_root)
        external = is_external_url(href)
        attributes = ' target="_blank" rel="noreferrer"' if external else ""
        rendered = (
            f'<a href="{escape(href, quote=True)}"{attributes}>'
            f"{format_nonlink_inline(label)}"
            "</a>"
        )
        token = f"@@PAPER_LINK_{len(links)}@@"
        links.append(rendered)
        return token

    protected = LINK_RE.sub(protect_link, text)
    rendered = format_nonlink_inline(protected)
    for index, link in enumerate(links):
        rendered = rendered.replace(f"@@PAPER_LINK_{index}@@", link)
    return rendered


def format_nonlink_inline(text: str) -> str:
    rendered = escape(text)
    rendered = STRONG_RE.sub(r"<strong>\1</strong>", rendered)
    rendered = CODE_RE.sub(r"<code>\1</code>", rendered)
    rendered = MATH_RE.sub(r'<span class="math">\1</span>', rendered)
    return rendered


def is_block_start(line: str) -> bool:
    return bool(
        not line.strip()
        or line.startswith("```")
        or line.startswith(">")
        or line.startswith("|")
        or HEADING_RE.match(line)
        or LIST_RE.match(line)
        or HORIZONTAL_RULE_RE.match(line)
    )


def render_table(lines: list[str], source_dir: Path, source_root: Path) -> str:
    rows: list[list[str]] = []
    for line in lines:
        if TABLE_SEPARATOR_RE.match(line):
            continue
        stripped = line.strip().strip("|")
        rows.append([cell.strip() for cell in stripped.split("|")])
    if not rows:
        return ""

    header = rows[0]
    body = rows[1:]
    header_html = "".join(f"<th>{inline_html(cell, source_dir, source_root)}</th>" for cell in header)
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{inline_html(cell, source_dir, source_root)}</td>" for cell in row)
        + "</tr>"
        for row in body
    )
    return f'<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>'


def render_list(lines: list[str], start: int, source_dir: Path, source_root: Path) -> tuple[str, int]:
    first = LIST_RE.match(lines[start])
    assert first is not None
    ordered = bool(first.group(3))
    tag = "ol" if ordered else "ul"
    items: list[str] = []
    index = start

    while index < len(lines):
        match = LIST_RE.match(lines[index])
        if not match or bool(match.group(3)) != ordered:
            break
        item_parts = [match.group(4)]
        index += 1
        while index < len(lines):
            continuation = lines[index]
            if not continuation.strip():
                next_line = lines[index + 1] if index + 1 < len(lines) else ""
                if LIST_RE.match(next_line):
                    index += 1
                    break
                break
            if LIST_RE.match(continuation):
                break
            if continuation.startswith((" ", "\t")):
                item_parts.append(continuation.strip())
                index += 1
                continue
            break
        items.append(f"<li>{inline_html(' '.join(item_parts), source_dir, source_root)}</li>")
        if index < len(lines) and not LIST_RE.match(lines[index]):
            break
    return f"<{tag}>{''.join(items)}</{tag}>", index


def render_blocks(lines: Iterable[str], source_dir: Path, source_root: Path) -> str:
    source_lines = list(lines)
    output: list[str] = []
    index = 0

    while index < len(source_lines):
        line = source_lines[index]
        if not line.strip():
            index += 1
            continue

        if line.startswith("```"):
            fence = line.strip()
            language = escape(fence[3:].strip())
            code_lines: list[str] = []
            index += 1
            while index < len(source_lines) and not source_lines[index].startswith("```"):
                code_lines.append(source_lines[index])
                index += 1
            if index < len(source_lines):
                index += 1
            class_attr = f' class="language-{language}"' if language else ""
            output.append(f"<pre><code{class_attr}>{escape(chr(10).join(code_lines))}</code></pre>")
            continue

        if line.strip() == "$$":
            math_lines: list[str] = []
            index += 1
            while index < len(source_lines) and source_lines[index].strip() != "$$":
                math_lines.append(source_lines[index])
                index += 1
            if index < len(source_lines):
                index += 1
            output.append(f'<div class="display-math">{escape(" ".join(math_lines).strip())}</div>')
            continue

        if line.startswith("|"):
            table_lines: list[str] = []
            while index < len(source_lines) and source_lines[index].startswith("|"):
                table_lines.append(source_lines[index])
                index += 1
            output.append(render_table(table_lines, source_dir, source_root))
            continue

        if line.startswith(">"):
            quote_lines: list[str] = []
            while index < len(source_lines) and source_lines[index].startswith(">"):
                quote_lines.append(source_lines[index].lstrip(">").strip())
                index += 1
            quote_text = " ".join(part for part in quote_lines if part)
            output.append(f'<aside class="source-note">{inline_html(quote_text, source_dir, source_root)}</aside>')
            continue

        if HORIZONTAL_RULE_RE.match(line):
            output.append("<hr>")
            index += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = min(max(len(heading.group(1)), 2), 5)
            title = inline_html(heading.group(2), source_dir, source_root)
            output.append(f"<h{level}>{title}</h{level}>")
            index += 1
            continue

        if LIST_RE.match(line):
            rendered_list, index = render_list(source_lines, index, source_dir, source_root)
            output.append(rendered_list)
            continue

        paragraph: list[str] = []
        while index < len(source_lines) and not is_block_start(source_lines[index]):
            paragraph.append(source_lines[index].strip())
            index += 1
        if paragraph:
            output.append(f"<p>{inline_html(' '.join(paragraph), source_dir, source_root)}</p>")
        else:
            index += 1

    return "\n".join(output)


def split_h2_sections(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    intro: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = H2_RE.match(line)
        if match:
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = match.group(1)
            current_lines = []
        elif current_heading is None:
            if not line.startswith("# "):
                intro.append(line)
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, current_lines))
    return intro, sections


@dataclass
class PaperSegment:
    """One semantic part of a paper note, normalized across the 7 note formats."""

    title: str
    kind: str
    lines: list[str]
    split_questions_from_lists: bool = False


@dataclass(frozen=True)
class BackgroundConcept:
    """A self-contained Korean explanation that can be reused across papers."""

    title: str
    aliases: tuple[str, ...]
    definition: str
    intuition: str
    why_it_matters: str
    code_connection: str
    common_confusion: str


def background_concept(
    title: str,
    aliases: tuple[str, ...],
    definition: str,
    intuition: str,
    why_it_matters: str,
    code_connection: str,
    common_confusion: str,
) -> BackgroundConcept:
    return BackgroundConcept(
        title,
        aliases,
        definition,
        intuition,
        why_it_matters,
        code_connection,
        common_confusion,
    )


# The notes intentionally name only the prerequisites.  This library turns
# those names into a usable, offline mini textbook.  A card is shown only when
# a paper's own prerequisite text contains one of its aliases.
BACKGROUND_CONCEPTS: tuple[BackgroundConcept, ...] = (
    background_concept(
        "텐서·shape·channel",
        ("tensor", "shape", "channel", "batch", "feature tensor", "tensor concatenation", "hidden state"),
        "텐서는 여러 축을 가진 수의 배열입니다. 이미지 배치는 보통 [B, C, H, W], 문장 표현은 [B, T, D]처럼 쓰며, B는 묶음 수, C/D는 특징 수, H·W/T는 공간 또는 토큰 축입니다.",
        "각 축이 무엇을 뜻하는지 읽으면 식이 훨씬 단순해집니다. 예를 들어 [32, 128]은 32개 예제마다 길이 128의 벡터 하나가 있다는 뜻입니다.",
        "논문의 결합, attention, convolution, pooling은 모두 어느 축을 섞고 어느 축을 보존하는지로 이해할 수 있습니다. 그래서 방법 절의 화살표보다 shape 변화를 먼저 따라가면 됩니다.",
        "코드에서는 tensor.shape, reshape/view, permute, cat/stack을 먼저 확인하세요. 오프라인 함수 사전에서 입력·출력 shape와 dtype·device를 함께 볼 수 있습니다.",
        "채널 수는 이미지의 색 채널만 뜻하지 않습니다. 중간층에서는 모델이 학습한 특징의 개수이며, batch 축과 섞어 해석하면 안 됩니다.",
    ),
    background_concept(
        "자동미분·gradient·역전파",
        ("gradient", "역전파", "backprop", "autodiff", "gradient flow", "log-derivative", "미분"),
        "gradient는 손실을 조금 줄이려면 각 파라미터를 어느 방향으로 얼마나 움직여야 하는지를 나타내는 미분값입니다. 역전파는 계산 그래프의 출력 쪽 손실에서 시작해 각 연산의 미분을 거꾸로 곱하여 gradient를 구하는 절차입니다.",
        "온도를 맞추는 손잡이가 많은 기계를 떠올리면 됩니다. 손실이 낮아지는 쪽으로 모든 손잡이를 아주 조금씩 돌리되, 연결된 부품일수록 영향이 연쇄적으로 전달됩니다.",
        "학습 가능 여부, residual connection, normalization, stop-gradient 같은 표현은 결국 gradient가 어디를 통과하고 어디에서 끊기는지를 설명합니다.",
        "PyTorch에서는 loss.backward()가 gradient를 누적하고 optimizer.step()이 파라미터를 갱신합니다. 학습 중에는 requires_grad와 model.train()/eval()도 함께 확인합니다.",
        "gradient가 작다는 것은 모델 출력값이 작다는 뜻이 아닙니다. 손실 표면이 평평하거나 경로가 길어 신호가 약해졌다는 뜻일 수 있습니다.",
    ),
    background_concept(
        "logit·softmax·교차 엔트로피",
        ("softmax", "logit", "cross-entropy", "cross entropy", "교차 엔트로피", "binary logistic", "BCE", "classification loss"),
        "logit은 확률로 바꾸기 전 모델의 원시 점수이고, softmax는 여러 logit을 합이 1인 확률 분포로 바꿉니다. 교차 엔트로피는 정답 분포와 모델 분포가 얼마나 다른지 재는 손실이라 정답 클래스 확률을 높이도록 학습시킵니다.",
        "세 클래스 점수 [2, 1, 0]은 누가 가장 그럴듯한지만 말합니다. softmax를 거치면 [0.67, 0.24, 0.09]처럼 확률로 읽을 수 있고, 정답이 첫 클래스라면 그 0.67을 1에 가깝게 만드는 것이 목표입니다.",
        "분류 결과, contrastive objective, 지식 증류의 온도 조절은 모두 점수에서 확률 또는 상대적 선호를 만드는 이 흐름 위에 있습니다.",
        "분류 코드에서는 F.cross_entropy(logits, target)를 주로 씁니다. 이 함수는 내부에서 안정적인 log-softmax를 처리하므로 보통 softmax 결과를 다시 넣지 않습니다.",
        "softmax와 cross-entropy 사이에 softmax를 두 번 적용하면 학습 신호가 약해질 수 있습니다. 이진 분류의 BCEWithLogitsLoss도 같은 원칙으로 logit을 직접 받습니다.",
    ),
    background_concept(
        "확률·우도·기댓값·Monte Carlo",
        ("likelihood", "우도", "probability", "확률", "Monte Carlo", "expectation", "기댓값", "Jensen", "sampling"),
        "확률 모델은 관측 데이터가 생길 가능성을 분포로 표현합니다. 우도는 고정된 데이터가 주어졌을 때 현재 파라미터가 그 데이터를 얼마나 잘 설명하는지이고, 기댓값은 분포에서 평균적으로 기대되는 값을 뜻합니다.",
        "복잡한 평균을 종이 위에서 정확히 계산하기 어려우면, 분포에서 여러 번 뽑아 평균을 내어 근사합니다. 이것이 Monte Carlo 추정이며 샘플 수가 늘수록 보통 안정적입니다.",
        "생성 모델의 목표함수, 정책 gradient의 기대 보상, 변분추론의 ELBO는 모두 ‘모든 가능한 경우의 평균’을 효율적으로 추정하는 문제입니다.",
        "코드에서는 torch.randn이나 분포의 sample로 표본을 만들고 mean으로 평균을 근사합니다. 시드 고정과 표본 수는 재현성과 분산에 직접 영향을 줍니다.",
        "확률이 높다고 해서 결과가 반드시 발생하는 것은 아닙니다. 하나의 표본과 분포 전체의 성질을 구분해야 합니다.",
    ),
    background_concept(
        "KL divergence·ELBO·변분추론",
        ("KL divergence", "KL", "ELBO", "variational", "변분", "Jensen inequality", "Jensen 부등식", "latent variable"),
        "KL divergence는 한 분포를 다른 분포로 근사할 때 생기는 정보 손실을 재는 비대칭 거리입니다. ELBO는 직접 계산하기 어려운 로그 우도의 하한으로, 데이터 재구성 품질과 잠재분포의 규칙성을 함께 최적화하도록 만듭니다.",
        "VAE에서는 encoder가 ‘이 입력이 어떤 잠재벡터일까’라는 근사 분포를 만들고, decoder가 그 벡터로 입력을 복원합니다. ELBO는 잘 복원하면서도 잠재공간이 뒤죽박죽되지 않게 균형을 맞춥니다.",
        "생성 모델, 지식 증류, 분포 정렬에서 KL은 단순한 오차가 아니라 두 확률 분포의 모양을 맞추는 언어입니다.",
        "코드에서는 평균·로그분산으로 Gaussian KL을 계산하거나 F.kl_div를 사용합니다. reduction 방식과 log-probability 입력 여부를 반드시 확인하세요.",
        "KL은 대칭적인 거리도, 항상 유한한 값도 아닙니다. 어느 분포를 기준으로 두느냐에 따라 벌점의 성격이 달라집니다.",
    ),
    background_concept(
        "정규화·BatchNorm·LayerNorm",
        ("normalization", "batch normalization", "batch norm", "LayerNorm", "LRN", "RMSNorm", "pre-norm", "정규화"),
        "정규화는 활성값의 평균과 크기를 일정한 범위로 조정해 학습을 안정시키는 기법입니다. BatchNorm은 미니배치 통계를, LayerNorm은 한 예제의 특징 축 통계를 주로 이용합니다.",
        "각 층의 신호 크기가 제각각이면 다음 층은 계속 움직이는 표적을 맞춰야 합니다. 정규화는 신호의 기준선을 맞춰 다음 층이 더 안정적으로 학습하게 합니다.",
        "깊은 CNN의 학습, Transformer의 residual block, 저비트 양자화에서 값의 분포를 다루는 설명을 이해하는 핵심입니다.",
        "코드에서는 nn.BatchNorm2d와 nn.LayerNorm이 대표적입니다. train/eval 모드에서 BatchNorm이 쓰는 통계가 달라지는 점을 실습으로 확인할 수 있습니다.",
        "정규화는 데이터를 0~1로 스케일링하는 전처리와 같은 말이 아닙니다. 어디의 통계를 어떤 축으로 계산하는지가 다릅니다.",
    ),
    background_concept(
        "임베딩·내적·코사인 유사도",
        ("embedding", "임베딩", "cosine", "dot product", "inner product", "distributional hypothesis", "representation", "표현"),
        "임베딩은 단어·이미지·노드처럼 이산적인 대상을 길이 D의 연속 벡터로 바꾼 표현입니다. 내적과 코사인 유사도는 두 벡터가 비슷한 방향이나 관계를 갖는지를 수치로 나타냅니다.",
        "‘고양이’와 ‘강아지’가 비슷한 문맥에서 쓰이면 벡터 공간에서 가까워지도록 학습할 수 있습니다. 코사인은 길이를 무시하고 방향만 비교하므로 크기보다 의미적 방향을 보고 싶을 때 유용합니다.",
        "검색, 대조학습, 추천, attention은 대부분 표현을 만들고 그 사이의 유사도를 비교하는 두 단계로 구성됩니다.",
        "코드에서는 F.normalize 뒤 행렬곱으로 코사인 유사도를 한꺼번에 계산하거나 F.cosine_similarity를 씁니다. [B, D]와 [N, D]의 축 방향을 확인하세요.",
        "내적이 큰 것이 항상 ‘의미가 더 비슷함’을 뜻하지는 않습니다. 벡터 길이까지 반영하는 내적과 방향만 보는 코사인의 차이를 구분해야 합니다.",
    ),
    background_concept(
        "합성곱·kernel·feature map",
        ("convolution", "합성곱", "kernel", "feature map", "stacked convolution", "standard convolution"),
        "2차원 합성곱은 작은 kernel을 이미지 위로 이동시키며 주변 픽셀 패턴과의 반응을 계산하는 연산입니다. 여러 kernel의 출력이 feature map이며, 각 채널은 모서리·질감·부분 모양 같은 서로 다른 특징에 반응하도록 학습됩니다.",
        "3×3 kernel을 작은 돋보기라고 생각하면 됩니다. 돋보기가 이미지 전체를 훑으며 같은 모양을 어디서 발견했는지 지도로 남깁니다.",
        "CNN 논문에서 layer를 쌓는다는 말은 원시 픽셀에서 시작해 점점 넓고 추상적인 패턴을 만드는 feature map 변환을 반복한다는 뜻입니다.",
        "PyTorch에서는 nn.Conv2d(in_channels, out_channels, kernel_size)를 봅니다. 입력 [B, C, H, W]에서 out_channels가 새 feature map의 수가 됩니다.",
        "딥러닝 구현의 convolution은 보통 kernel을 뒤집지 않는 cross-correlation을 사용하지만 관습적으로 convolution이라고 부릅니다.",
    ),
    background_concept(
        "padding·stride·receptive field",
        ("padding", "stride", "receptive field", "valid padding", "same padding", "dilation"),
        "padding은 경계에 값을 덧대 공간 크기를 조절하고, stride는 kernel을 몇 칸씩 건너뛸지 정합니다. receptive field는 한 출력 위치가 원본 입력의 어느 범위를 볼 수 있는지입니다.",
        "stride를 2로 늘리면 지도 위의 점을 절반 간격으로 찍는 것처럼 feature map이 작아집니다. 층을 쌓을수록 한 점이 보는 원본 영역은 넓어집니다.",
        "모델이 작은 질감만 보는지, 물체 전체 문맥까지 보는지를 설명할 때 receptive field가 쓰입니다. 탐지·분할처럼 위치가 중요한 과제에서는 크기 변화도 함께 읽어야 합니다.",
        "Conv2d의 kernel_size, stride, padding 인자가 shape를 바꿉니다. 계산한 shape와 tensor.shape를 비교해 보는 것이 가장 빠른 이해 방법입니다.",
        "receptive field가 크다고 해서 실제로 모든 픽셀을 똑같이 활용한다는 뜻은 아닙니다. 이론적 범위와 학습된 유효 영향 범위는 다를 수 있습니다.",
    ),
    background_concept(
        "pooling·downsampling·global average pooling",
        ("pooling", "subsampling", "downsampling", "global average pooling", "max pooling", "average pooling"),
        "pooling은 작은 창 안의 값을 max 또는 평균으로 요약해 공간 해상도를 줄이는 연산입니다. global average pooling은 각 채널의 H×W 전체를 평균내어 채널당 숫자 하나를 만듭니다.",
        "사진을 멀리서 볼수록 세부 위치는 덜 보이지만 큰 형태는 남습니다. pooling은 이런 요약을 통해 계산량을 줄이고 약간의 위치 변화에 둔감하게 합니다.",
        "CNN의 크기 축소, 가벼운 분류 head, receptive field 확대를 이해할 때 필요합니다.",
        "nn.MaxPool2d, nn.AvgPool2d, AdaptiveAvgPool2d가 대표적입니다. pooling 전후 [H, W]가 어떻게 바뀌는지 먼저 확인하세요.",
        "pooling은 항상 좋은 것이 아닙니다. 작은 물체나 경계처럼 정확한 위치가 필요한 task에서는 너무 이른 축소가 정보를 잃게 합니다.",
    ),
    background_concept(
        "채널 projection·1×1 convolution·branch 결합",
        ("1×1 convolution", "channel projection", "branch", "concatenation", "concat", "bottleneck", "channel split", "channel shuffle"),
        "1×1 convolution은 각 공간 위치에서 채널 벡터만 섞는 선형 변환입니다. projection은 특징 수를 줄이거나 맞추고, branch 결합은 여러 경로의 특징을 concat 또는 더하기로 합칩니다.",
        "한 픽셀 위치의 채널 64개를 16개로 압축하는 작은 전용 mixing table이라고 보면 됩니다. 공간 문양을 새로 보지 않고 ‘어떤 특징을 함께 쓸지’를 재조합합니다.",
        "Inception, bottleneck, 경량 CNN은 계산량과 표현력을 조절하기 위해 이 구조를 적극적으로 씁니다.",
        "코드에서는 Conv2d(..., kernel_size=1), torch.cat(..., dim=채널축), residual add를 만납니다. concat은 채널 수를 늘리고 add는 같은 shape를 요구합니다.",
        "1×1은 아무 일도 하지 않는 연산이 아닙니다. 공간 창은 작지만 channel mixing과 비선형성 조합이 강력한 표현 변환을 만듭니다.",
    ),
    background_concept(
        "encoder-decoder·U-Net·skip connection",
        ("encoder-decoder", "encoder decoder", "U-Net", "upsampling", "transposed convolution", "skip connection", "mask decoder"),
        "encoder는 입력을 더 작고 추상적인 표현으로 압축하고, decoder는 그 표현을 목표 출력의 해상도로 되돌립니다. U-Net의 skip connection은 encoder의 고해상도 특징을 decoder에 바로 전달해 경계와 위치 정보를 보존합니다.",
        "전체 장면을 이해하려면 멀리서 보는 encoder가 필요하지만, 픽셀 단위 결과에는 원래의 세밀한 윤곽도 필요합니다. skip은 이 두 종류의 정보를 연결합니다.",
        "분할, depth, 복원, 생성처럼 입력과 출력 모두 공간 구조를 갖는 논문의 기본 골격입니다.",
        "코드에서는 ConvTranspose2d 또는 interpolate로 크기를 키우고, encoder feature와 concat합니다. 크기가 한 픽셀씩 어긋나면 crop 또는 padding 처리의 이유를 찾아보세요.",
        "transposed convolution은 단순히 convolution을 거꾸로 계산하는 것이 아닙니다. 출력 크기를 키우는 학습 가능한 연산이며 checkerboard artifact가 생길 수도 있습니다.",
    ),
    background_concept(
        "residual connection·identity mapping",
        ("residual", "identity mapping", "shortcut connection", "projection shortcut", "residual function"),
        "residual block은 원하는 변환 H(x)를 직접 만들기보다 입력 x에 작은 변화 F(x)를 더해 H(x)=x+F(x) 형태로 학습합니다. shape가 다르면 projection으로 입력 경로의 채널이나 크기를 맞춥니다.",
        "기존 답을 전부 다시 쓰는 대신 ‘무엇을 고칠지’만 적는 방식입니다. 필요 없으면 F(x)를 0에 가깝게 두어 원래 정보를 안전하게 통과시킬 수 있습니다.",
        "매우 깊은 CNN과 Transformer가 안정적으로 학습되는 이유, gradient flow, pre-norm 구조를 읽는 핵심 열쇠입니다.",
        "코드에서는 y = x + block(x)가 기본이며, channel/stride가 달라지면 1×1 convolution projection을 둡니다.",
        "skip connection과 residual add는 모두 지름길이지만 concat 기반 U-Net skip처럼 채널을 늘리는 연결과는 shape·의미가 다릅니다.",
    ),
    background_concept(
        "semantic segmentation·pixel-wise loss·Dice",
        ("semantic segmentation", "pixel-wise", "Dice", "dense prediction", "mask", "segmentation"),
        "semantic segmentation은 이미지의 각 픽셀에 클래스를 붙이는 과제입니다. pixel-wise loss는 각 위치의 예측을 정답 mask와 비교하고, Dice는 예측 영역과 정답 영역이 얼마나 겹치는지를 직접 재는 지표·손실입니다.",
        "사진 전체에 고양이가 있느냐가 아니라, 어느 픽셀이 고양이인가를 칠하는 일입니다. 작은 객체나 클래스 불균형에서는 단순 정확도보다 겹침 비율이 더 유용할 수 있습니다.",
        "U-Net, SAM, depth 같은 dense output 논문에서 output의 공간 축을 잃지 말아야 하는 이유를 설명합니다.",
        "출력은 보통 [B, classes, H, W]이고 정답 mask는 [B, H, W] 또는 one-hot 형태입니다. BCE, cross entropy, Dice의 입력 형식을 구분하세요.",
        "Dice가 높다고 모든 경계가 정확한 것은 아닙니다. 큰 배경이 많은 데이터에서는 클래스별 지표와 시각화도 함께 봐야 합니다.",
    ),
    background_concept(
        "물체 탐지·box·IoU·anchor",
        ("bounding box", "box regression", "IoU", "intersection over union", "anchor", "objectness", "detection", "one-stage", "two-stage", "Smooth-L1"),
        "물체 탐지는 클래스뿐 아니라 사각형 위치를 맞히는 과제입니다. IoU는 예측 box와 정답 box의 교집합을 합집합으로 나눈 값이며, anchor는 여러 크기·비율의 후보 box를 미리 놓아 두는 방식입니다.",
        "정답 상자를 겹쳐 그려 보고, 겹치는 면적이 전체 면적에서 얼마나 되는지 생각하면 IoU를 직관적으로 이해할 수 있습니다. objectness는 그 상자에 어떤 물체라도 있을 가능성입니다.",
        "Faster R-CNN, YOLO 계열의 proposal·loss·평가를 읽는 출발점이며, DETR가 anchor 없이 set prediction을 쓰는 이유와도 연결됩니다.",
        "box는 xyxy 또는 중심좌표+너비·높이로 표현합니다. 형식과 정규화 여부가 다르면 IoU 계산이 즉시 틀어집니다.",
        "IoU가 높은 box가 항상 올바른 클래스는 아닙니다. 위치 회귀와 분류/objectness는 별도의 신호입니다.",
    ),
    background_concept(
        "ViT patch·class token·positional embedding",
        ("image patch", "patch token", "class token", "positional embedding", "ViT", "position embedding", "RoPE"),
        "Vision Transformer는 이미지를 작은 patch로 나누고, 각 patch를 벡터 token으로 바꿔 Transformer에 넣습니다. class token은 전체 이미지 정보를 모으는 특별한 token이고, positional embedding은 순서가 없는 attention에 위치 정보를 알려 줍니다.",
        "문장을 단어 조각으로 읽듯, 이미지를 정사각형 조각의 문장으로 읽는다고 생각하면 됩니다. token만 보면 조각의 순서를 모르므로 위치 표지가 필요합니다.",
        "ViT, CLIP, MAE, DETR 후속 모델에서 이미지와 텍스트를 같은 token 처리 방식으로 다루는 이유를 설명합니다.",
        "코드에서는 [B, C, H, W]를 patch embedding으로 [B, N, D]로 바꾸고 class token을 앞에 붙입니다. N은 patch 개수입니다.",
        "class token이 항상 필수는 아닙니다. global average pooling이나 다른 pooling token을 쓰는 구조도 있습니다.",
    ),
    background_concept(
        "attention·Q/K/V·multi-head·mask",
        ("attention", "query/key/value", "Q/K/V", "multi-head", "self-attention", "causal mask", "masked attention", "cross-attention", "scaled dot-product"),
        "attention은 query가 key들과의 관련도를 계산해 value를 가중 평균하는 연산입니다. scaled dot-product attention은 softmax(QKᵀ/√d)V로 쓰며, multi-head는 서로 다른 관점의 attention을 병렬로 학습합니다.",
        "질문(query)을 들고 색인(key)을 훑은 뒤, 관련 문서의 내용(value)을 비율대로 섞는 과정입니다. mask는 볼 수 없는 위치의 점수를 막아 미래 토큰이나 padding을 참고하지 못하게 합니다.",
        "Transformer, graph attention, DETR query, cross-modal 모델에서 ‘무엇이 무엇을 참고하는가’를 읽는 핵심 연산입니다.",
        "코드에서는 Q·K·V shape가 보통 [B, heads, T, d_head]이고, score의 마지막 두 축이 [query 길이, key 길이]입니다. mask의 broadcast 방향을 확인하세요.",
        "attention weight가 항상 설명 가능한 인과 근거는 아닙니다. 모델이 참고한 패턴의 한 표현일 뿐, 단독 설명으로 과해석하면 안 됩니다.",
    ),
    background_concept(
        "DETR object query·Hungarian matching",
        ("object query", "Hungarian", "bipartite", "set prediction", "no-object", "generalized IoU"),
        "DETR은 고정 개수의 learned object query가 이미지 특징을 읽어 각각 하나의 물체 후보를 예측하게 합니다. Hungarian matching은 예측 집합과 정답 집합을 비용이 가장 작게 되도록 일대일로 짝지어 중복 예측을 줄입니다.",
        "학생 여러 명에게 서로 다른 문제를 하나씩 배정하듯, 어떤 query가 어떤 정답 물체를 맡을지 전체 조합을 보고 결정합니다. 남는 query는 no-object로 학습됩니다.",
        "anchor와 NMS 중심 detector에서 set-based detector로 넘어가는 논문의 loss와 학습 안정성 설명에 필요합니다.",
        "코드에서는 분류 비용과 box L1·IoU 비용 행렬을 만든 뒤 assignment를 구합니다. batch마다 정답 물체 수가 달라지는 점을 신경 써야 합니다.",
        "Hungarian matching은 추론 단계의 NMS가 아닙니다. 주로 학습 중 예측과 정답의 대응을 정하는 절차입니다.",
    ),
    background_concept(
        "카메라·투영·3D point cloud",
        ("camera projection", "camera intrinsic", "camera extrinsic", "point cloud", "multi-view", "depth", "NeRF", "differentiable rendering", "Gaussian splatting"),
        "카메라 내참수(intrinsic)는 초점거리·주점처럼 렌즈와 영상 좌표의 성질을, 외참수(extrinsic)는 세계 좌표에서 카메라의 위치와 방향을 나타냅니다. 3D 점은 이 두 변환을 거쳐 2D 이미지 위치로 투영됩니다.",
        "세계 속 점을 카메라 좌표계로 옮긴 뒤 화면에 그림자처럼 찍는 과정입니다. 여러 시점에서 같은 점을 보면 깊이와 카메라 위치를 함께 추정할 수 있습니다.",
        "3D 재구성, novel-view rendering, monocular depth, point tracking 논문에서 데이터와 output이 어떤 좌표계에 있는지 이해하게 해 줍니다.",
        "코드에서는 homogeneous coordinate, [R|t], K 행렬, depth map을 만납니다. 픽셀 좌표와 정규화 카메라 좌표를 혼동하지 마세요.",
        "한 장의 이미지로 깊이를 추정하는 것은 본질적으로 모호합니다. 학습 데이터의 prior가 이 모호성을 줄여 주는 역할을 합니다.",
    ),
    background_concept(
        "토큰화·seq2seq·teacher forcing",
        ("tokenization", "token", "seq2seq", "teacher forcing", "encoder decoder", "causal language modeling", "autoregressive"),
        "토큰화는 텍스트를 모델이 처리할 정수 ID 조각으로 나누는 과정입니다. seq2seq는 입력 시퀀스를 받아 출력 시퀀스를 만들고, teacher forcing은 학습 중 이전 단계의 정답 토큰을 decoder 입력으로 주는 방식입니다.",
        "번역을 배울 때 정답 문장의 앞부분을 보며 다음 단어를 맞히게 하는 것과 같습니다. 추론 때는 정답이 없으므로 모델 자신의 이전 출력으로 이어 갑니다.",
        "RNN·Transformer 번역, 언어 모델, RAG의 생성 단계에서 train과 inference의 차이를 읽는 데 필요합니다.",
        "코드에서는 input_ids의 shape가 [B, T]이고, language-model logits는 [B, T, vocab]인 경우가 많습니다. pad token과 attention mask도 같이 다룹니다.",
        "teacher forcing이 학습을 쉽게 해도 추론 오류가 누적되는 exposure bias를 완전히 없애지는 않습니다.",
    ),
    background_concept(
        "RNN·LSTM hidden state",
        ("RNN", "LSTM", "hidden state", "recurrent", "cell state"),
        "RNN은 토큰을 한 단계씩 읽으며 이전 상태 h를 다음 상태로 갱신합니다. LSTM은 gate와 cell state를 추가해 오래 전 정보가 사라지는 문제를 완화한 순환 구조입니다.",
        "문장을 읽으면서 지금까지의 메모를 갱신하는 작은 노트라고 생각하면 됩니다. hidden state는 그 시점까지 읽은 내용을 압축한 메모입니다.",
        "word2vec 이후의 문맥 모델, seq2seq, attention이 왜 등장했는지를 이해하는 역사적 기준점입니다.",
        "PyTorch RNN/LSTM 출력은 보통 [B, T, hidden] 또는 [T, B, hidden]입니다. batch_first 설정과 마지막 hidden state의 의미를 확인하세요.",
        "LSTM이 모든 긴 문맥을 완벽히 기억하는 것은 아닙니다. 병렬 처리와 긴 의존성 측면에서 Transformer가 다른 장점을 가집니다.",
    ),
    background_concept(
        "사전학습·fine-tuning·LoRA/PEFT",
        ("pretrain", "pre-training", "fine-tune", "fine tuning", "LoRA", "low-rank", "PEFT", "instruction tuning", "frozen backbone"),
        "사전학습은 대규모 일반 데이터에서 재사용 가능한 표현을 먼저 배우는 과정이고, fine-tuning은 특정 과제 데이터에 맞게 일부 또는 전체 파라미터를 조정하는 과정입니다. LoRA는 큰 가중치를 그대로 두고 작은 저랭크 보정 행렬만 학습하는 PEFT 기법입니다.",
        "이미 숙련된 모델을 새 과제에 맞춰 미세 조정하는 일입니다. LoRA는 원본 책 전체를 다시 쓰지 않고 여백에 얇은 보정 노트를 붙이는 방식에 가깝습니다.",
        "LLM·비전 foundation model의 비용, 적응, frozen encoder와 가벼운 adapter의 trade-off를 읽는 데 필요합니다.",
        "코드에서는 requires_grad=False로 backbone을 고정하고 adapter 파라미터만 optimizer에 넘기는 흐름을 볼 수 있습니다. rank r이 작을수록 학습량은 줄지만 표현력도 제한됩니다.",
        "fine-tuning은 항상 전체 모델 학습보다 낫지 않습니다. 데이터 규모, 도메인 차이, 망각 위험에 따라 선택이 달라집니다.",
    ),
    background_concept(
        "dense retrieval·MIPS·RAG",
        ("retrieval", "dense retrieval", "MIPS", "maximum inner product", "RAG", "latent document", "recall@k", "retriever"),
        "dense retrieval은 질의와 문서를 같은 임베딩 공간에 놓고 내적 또는 코사인 점수가 큰 문서를 찾는 방식입니다. RAG는 찾은 문서를 생성 모델의 문맥으로 넣어, 답을 만들 때 외부 근거를 활용하게 합니다.",
        "시험 중 기억만으로 답하지 않고, 먼저 도서관에서 관련 쪽을 찾아 책상 위에 펼친 뒤 답안을 쓰는 과정입니다. MIPS는 그중 질의와 가장 잘 맞는 벡터를 빠르게 찾는 문제입니다.",
        "RAG 논문의 성능은 생성 모델만이 아니라 retriever의 recall, 문서 분할, prompt 구성에 크게 좌우된다는 점을 이해하게 해 줍니다.",
        "이 프로젝트의 TfidfRetriever와 DenseSemanticRetriever에서 query vector, document matrix, top-k score를 직접 확인할 수 있습니다. Recall@k는 정답 문서가 상위 k 안에 있는 비율입니다.",
        "검색 점수가 높아도 문서가 답을 충분히 뒷받침하지 않을 수 있습니다. retrieval 지표와 최종 답변의 사실성을 따로 평가해야 합니다.",
    ),
    background_concept(
        "autoencoder·VAE·reparameterization",
        ("autoencoder", "AE", "VAE", "variational autoencoder", "reparameterization", "encoder/decoder", "latent space", "잠재"),
        "autoencoder는 입력을 작은 latent 벡터로 압축한 뒤 다시 복원합니다. VAE는 latent를 하나의 점이 아니라 분포로 보고, 평균과 분산을 학습해 새로운 표본도 생성할 수 있게 합니다. reparameterization은 z=μ+σ·ε 형태로 무작위성을 분리해 gradient가 encoder까지 흐르게 하는 기법입니다.",
        "사진을 좌표 몇 개로 압축했다가 복원하는 방식입니다. VAE는 그 좌표 주변을 부드러운 지도처럼 만들어, 근처를 샘플링해도 그럴듯한 결과가 나오게 합니다.",
        "잠재변수 생성 모델의 목적함수, KL 항, 보간과 disentanglement 설명의 기반입니다.",
        "코드에서는 encoder가 mu와 logvar를 내고, epsilon=torch.randn_like(mu)로 샘플을 만듭니다. reconstruction loss와 KL loss의 크기를 따로 기록하세요.",
        "VAE의 latent가 정답 의미 축을 자동으로 깔끔히 분리한다는 보장은 없습니다. 목적함수 가중치와 데이터에 따라 결과가 크게 달라집니다.",
    ),
    background_concept(
        "GAN·minimax·판별기",
        ("GAN", "minimax", "discriminator", "generator", "adversarial", "Wasserstein", "Earth Mover", "Lipschitz"),
        "GAN은 가짜를 만드는 generator와 진짜·가짜를 구분하는 discriminator가 경쟁하며 학습하는 생성 모델입니다. minimax 목표는 generator가 discriminator를 속이고 discriminator가 이를 구별하려는 게임으로 표현됩니다.",
        "위조지폐 제작자와 감별사가 서로 실력을 높이는 과정처럼 생각하면 됩니다. 감별사가 좋아질수록 제작자도 더 그럴듯한 샘플을 만들어야 합니다.",
        "생성 품질, mode collapse, Wasserstein 거리와 Lipschitz 제약이 왜 등장하는지 이해하는 출발점입니다.",
        "코드에서는 generator와 discriminator를 번갈아 update하며, discriminator의 gradient를 generator update 때 끊습니다. 두 optimizer의 순서를 주의 깊게 읽으세요.",
        "GAN loss가 낮다고 이미지가 항상 좋아지는 것은 아닙니다. 두 모델의 균형이 무너지면 loss만으로 품질을 읽기 어렵습니다.",
    ),
    background_concept(
        "diffusion·Markov chain·noise schedule",
        ("diffusion", "Markov chain", "noise schedule", "variance schedule", "Gaussian noise", "denoising", "forward process"),
        "diffusion model은 데이터에 작은 Gaussian noise를 여러 단계로 더해 거의 순수한 잡음으로 만드는 forward process와, 그 잡음을 거꾸로 제거하는 reverse process를 학습합니다. Markov 성질은 다음 상태가 바로 이전 상태에만 의존한다는 뜻입니다.",
        "깨끗한 사진에 조금씩 눈을 뿌린 뒤, 어느 정도의 눈이 섞였는지를 알려 주면서 원래 사진을 복원하는 법을 배우는 과정입니다. schedule은 각 단계에 얼마나 노이즈를 넣을지 정합니다.",
        "DDPM, score model, latent diffusion의 학습 목표와 샘플링 속도·품질 trade-off를 읽는 기초입니다.",
        "코드에서는 timestep t, alpha_bar[t], noise ε를 만들고 모델이 noise 또는 velocity를 예측하게 합니다. 입력 image와 t embedding의 batch 축을 맞추세요.",
        "forward process가 학습되는 것이 아니라 보통 미리 정한 schedule이라는 점, 그리고 reverse process는 확률적 또는 결정론적일 수 있다는 점을 구분하세요.",
    ),
    background_concept(
        "score·SDE·Langevin dynamics",
        ("score", "Ito SDE", "SDE", "Euler–Maruyama", "Euler-Maruyama", "Langevin", "∇x log p", "score matching"),
        "score는 데이터 밀도 log p(x)의 입력에 대한 gradient ∇x log p(x)로, 현재 점에서 확률이 더 높은 곳으로 향하는 방향을 알려 줍니다. SDE는 연속 시간에서 drift와 noise가 함께 변하는 확률 미분방정식이고, Euler–Maruyama는 이를 작은 시간 간격으로 근사하는 방법입니다.",
        "안개 낀 산에서 가장 높은 곳으로 오르는 방향을 알려 주는 나침반이 score입니다. Langevin dynamics는 그 방향으로 조금 움직이되 작은 무작위 흔들림도 넣어 분포를 탐색합니다.",
        "score-based diffusion이 이산 DDPM을 연속 시간 관점으로 일반화하고, 다양한 sampler를 비교하는 언어를 제공합니다.",
        "실습에서는 score network 입력에 noise level을 함께 넣고, 작은 step으로 업데이트합니다. step size가 크면 수치적으로 불안정해질 수 있습니다.",
        "score 자체가 확률값은 아닙니다. 확률 밀도가 증가하는 방향을 나타내는 벡터장입니다.",
    ),
    background_concept(
        "MDP·trajectory·return",
        ("MDP", "trajectory", "return", "reward", "state", "action", "Monte Carlo return", "Markov decision"),
        "MDP는 상태 s에서 행동 a를 고르면 보상 r과 다음 상태 s'가 확률적으로 정해지는 순차 의사결정 틀입니다. trajectory는 한 에피소드의 상태·행동·보상 기록이고, return은 앞으로 받을 할인 보상의 합입니다.",
        "게임을 하며 현재 화면을 보고 버튼을 누르고 점수를 받는 반복입니다. 좋은 행동은 지금 점수만이 아니라 이후 점수까지 많이 얻는 행동입니다.",
        "강화학습 논문의 정책, 가치함수, 탐색, offline data가 모두 무엇을 최적화하는지 이해하는 공통 언어입니다.",
        "코드에서는 (state, action, reward, next_state, done) transition을 저장하고 G_t = r_t + γr_{t+1}+…를 계산합니다. done 이후에는 bootstrap을 끊어야 합니다.",
        "Markov 가정은 현재 상태가 미래에 필요한 정보를 충분히 담는다고 보는 모델링 가정입니다. 실제 관측이 불완전하면 POMDP 문제가 됩니다.",
    ),
    background_concept(
        "Bellman·Q-value·TD error",
        ("Bellman", "Q-value", "Q function", "TD error", "temporal difference", "optimality equation", "value function"),
        "Q(s,a)는 상태 s에서 행동 a를 한 뒤 얻을 미래 return의 기대값입니다. Bellman 방정식은 현재 가치가 즉시 보상과 다음 상태 가치의 조합이라는 재귀 관계이고, TD error는 현재 예측과 그 목표값의 차이입니다.",
        "먼 미래 점수를 한 번에 맞히기보다 ‘이번 보상 + 다음 장면의 예상 점수’로 매 단계의 추정을 고쳐 나가는 방식입니다.",
        "DQN, actor-critic, planning이 왜 bootstrap target을 만들고 target network를 쓰는지 설명합니다.",
        "코드에서는 target = r + gamma * max_a' Q_target(s', a')를 만들고 MSE 또는 Huber loss로 Q를 맞춥니다. terminal transition에는 다음 가치가 없습니다.",
        "Q 값은 확률도 즉시 보상도 아닙니다. 특정 정책·환경에서 기대되는 누적 보상입니다.",
    ),
    background_concept(
        "policy gradient·advantage·actor-critic",
        ("policy gradient", "actor-critic", "advantage", "REINFORCE", "baseline", "log-derivative trick", "critic"),
        "policy gradient는 행동을 고르는 확률 정책 πθ를 직접 미분해 기대 return을 키웁니다. actor는 행동 정책이고, critic은 가치 추정을 제공하며, advantage는 실제 return이 기준 가치보다 얼마나 좋았는지 나타내 분산을 줄입니다.",
        "좋은 결과를 낸 선택의 확률은 높이고 나쁜 결과를 낸 선택의 확률은 낮추되, 평소 기대치보다 얼마나 좋았는지를 기준으로 칭찬의 세기를 조절합니다.",
        "REINFORCE의 높은 분산, A2C/A3C/PPO의 안정화 장치, entropy bonus를 읽는 핵심입니다.",
        "코드에서는 log_prob * advantage의 평균에 음수를 붙여 loss로 최소화합니다. advantage를 detach하지 않으면 actor와 critic gradient가 섞일 수 있습니다.",
        "advantage가 양수라는 것은 그 행동 자체가 절대적으로 좋은 것이 아니라 현재 critic 기준보다 좋았다는 뜻입니다.",
    ),
    background_concept(
        "replay buffer·target network·epsilon-greedy",
        ("replay buffer", "target network", "epsilon-greedy", "experience replay", "off-policy"),
        "replay buffer는 과거 transition을 저장해 무작위 미니배치로 다시 학습하게 하는 메모리입니다. target network는 TD 목표를 만들 때 천천히 갱신되는 별도 Q 네트워크이고, epsilon-greedy는 대부분 최고 Q 행동을 고르되 일정 확률로 무작위 탐색을 합니다.",
        "최근 경험만 반복하면 연속 장면이 너무 비슷해 학습이 흔들립니다. 경험을 섞고, 답안지 역할의 target을 잠시 고정해 움직이는 목표를 완화합니다.",
        "DQN 계열의 안정성, exploration, offline·online 데이터 재사용을 설명합니다.",
        "코드에서는 deque에서 random sample을 뽑고, 일정 step마다 target.load_state_dict를 합니다. epsilon schedule과 buffer warm-up 시점을 로그로 확인하세요.",
        "target network는 정답 모델이 아닙니다. 학습 중인 online network의 지연된 복사본입니다.",
    ),
    background_concept(
        "PPO ratio·importance sampling·entropy",
        ("PPO", "importance ratio", "importance sampling", "clipping", "maximum-entropy", "entropy objective", "SAC"),
        "importance ratio는 새 정책이 과거 정책으로 수집한 행동을 얼마나 더 또는 덜 선호하는지의 비율입니다. PPO는 이 비율을 일정 범위에서 clip해 한 번의 update가 정책을 과도하게 바꾸지 못하게 하고, entropy 보너스는 너무 이른 확신을 막습니다.",
        "이전 정책으로 모은 경험을 재활용하되, 갑자기 성격이 완전히 다른 정책으로 바꾸지 않도록 안전 난간을 두는 방식입니다.",
        "PPO의 안정성, off-policy correction, SAC의 탐색 장려 목적을 읽는 데 필요합니다.",
        "코드에서는 ratio = exp(new_log_prob - old_log_prob)를 만들고 clipped surrogate의 min을 사용합니다. old_log_prob는 보통 gradient가 흐르지 않는 저장값입니다.",
        "clip은 성능 향상을 보장하는 마법이 아닙니다. advantage 추정, reward scale, batch 수가 함께 안정성에 영향을 줍니다.",
    ),
    background_concept(
        "MCTS·UCB·PUCT",
        ("MCTS", "UCB", "PUCT", "tree search", "AlphaZero"),
        "Monte Carlo Tree Search는 가능한 행동의 트리를 필요한 곳만 확장하며 rollout 또는 가치 예측을 이용해 탐색하는 방법입니다. UCB/PUCT는 많이 시도해 본 선택의 평균 가치와 덜 시도한 선택의 탐색 보너스를 함께 고려합니다.",
        "유망해 보이는 길을 더 자주 가되, 아직 가 보지 않은 길도 가끔 확인하는 여행 계획입니다. 방문 횟수와 가치가 다음 행동 확률을 만듭니다.",
        "AlphaGo/AlphaZero가 policy·value network와 planning을 결합하는 방식, self-play data 생성의 의미를 설명합니다.",
        "구현에서는 node의 visit count N, value sum W, prior P를 갱신합니다. leaf 평가와 backpropagation의 부호 전환을 특히 조심하세요.",
        "MCTS의 tree는 학습 신경망의 계산 그래프가 아닙니다. 환경 행동의 미래 분기를 나타내는 탐색 구조입니다.",
    ),
    background_concept(
        "인접행렬·degree·graph Laplacian",
        ("adjacency", "인접 행렬", "degree", "graph Laplacian", "Laplacian", "spectral gap"),
        "그래프는 node와 edge로 관계를 표현합니다. 인접행렬 A의 Aᵢⱼ는 i와 j가 연결됐는지 나타내고, degree는 node에 붙은 edge 수입니다. Laplacian은 연결 구조와 신호의 매끄러움을 분석하는 행렬로, spectral gap은 그래프의 연결성·혼합 성질과 관련됩니다.",
        "친구 관계표에서 행과 열은 사람이고, 표시된 칸이 연결입니다. 이 표를 행렬로 다루면 이웃의 정보가 어떻게 퍼지는지 계산할 수 있습니다.",
        "GCN의 정규화, random walk, expander graph, spectral 관점의 추천 논문을 읽는 기반입니다.",
        "코드에서는 edge_index 또는 sparse adjacency를 쓰며, degree가 큰 node의 영향이 과도해지지 않도록 normalization을 합니다.",
        "인접행렬이 대칭인 것은 무방향 그래프일 때뿐입니다. 추천·지식그래프 같은 방향 그래프에서는 Aᵢⱼ와 Aⱼᵢ가 다를 수 있습니다.",
    ),
    background_concept(
        "random walk·Skip-gram·negative sampling",
        ("random walk", "Markov", "Skip-gram", "negative sampling", "DeepWalk", "node2vec", "distributional"),
        "random walk는 현재 node에서 이웃으로 무작위 이동하는 과정입니다. DeepWalk는 walk로 만든 node 시퀀스를 문장처럼 보고 Skip-gram으로 주변 node를 예측해 임베딩을 학습합니다. negative sampling은 모든 node를 비교하는 대신 몇 개의 부정 예시만 뽑아 계산을 줄입니다.",
        "그래프 위를 산책하며 자주 같이 등장하는 node를 서로 가까운 단어처럼 학습하는 방식입니다.",
        "그래프 임베딩의 역사, 구조적 유사성과 이웃 근접성, GNN 이전의 관계 표현 방법을 이해하게 합니다.",
        "코드에서는 walk 길이·window size·negative 개수가 학습 데이터와 비용을 바꿉니다. 양성 쌍과 음성 쌍의 label 방향을 확인하세요.",
        "random walk에서 가까이 등장한다고 반드시 같은 라벨이라는 뜻은 아닙니다. 그래프의 관계 유형과 walk 편향에 따라 의미가 달라집니다.",
    ),
    background_concept(
        "GNN message passing·GCN·GAT",
        ("message passing", "GCN", "GAT", "graph convolution", "graph attention", "neighbor aggregation", "LightGCN"),
        "GNN은 각 node가 이웃 node의 표현을 모아 자신의 표현을 갱신하는 message passing을 반복합니다. GCN은 정규화된 이웃 평균을, GAT는 이웃별 중요도를 attention으로 학습합니다.",
        "사람마다 이웃의 소식을 받아 자신의 프로필을 갱신하는 과정입니다. 여러 번 반복하면 더 먼 이웃의 정보도 간접적으로 섞입니다.",
        "노드 분류, link prediction, 추천에서 그래프 구조를 feature에 반영하는 방법 절을 읽을 때 필요합니다.",
        "코드에서는 x와 edge_index가 기본 입력이며, layer 수가 늘면 k-hop 정보가 섞입니다. node 수와 edge 수의 축을 혼동하지 마세요.",
        "층을 너무 많이 쌓으면 node 표현이 서로 비슷해지는 over-smoothing이 생길 수 있습니다. 더 깊다고 항상 더 좋은 것은 아닙니다.",
    ),
    background_concept(
        "permutation invariance·1-WL·graph isomorphism",
        ("permutation invariance", "1-WL", "Weisfeiler", "graph isomorphism", "GIN", "homophily", "heterophily"),
        "그래프 node의 번호를 바꿔도 같은 그래프라면 모델 출력은 바뀌지 않아야 합니다. 1-WL test는 이웃 색을 반복적으로 모아 그래프를 구별하는 고전적 절차이며, 많은 message-passing GNN의 표현력 기준점입니다.",
        "사람 이름표를 바꿔도 친구 관계 자체는 같아야 합니다. 모델은 번호가 아니라 연결 구조와 특징을 봐야 합니다.",
        "GNN의 한계, subgraph GNN, positional encoding, heterophily 대응 방법을 논리적으로 읽게 해 줍니다.",
        "코드에서 node index를 permute해도 pool된 graph-level 출력이 유지되는지 작은 테스트로 확인할 수 있습니다.",
        "permutation invariance는 모든 node가 같은 출력을 내야 한다는 뜻이 아닙니다. 번호 바꾸기에 일관되게 함께 바뀌어야 한다는 뜻입니다.",
    ),
    background_concept(
        "link prediction·BPR·Recall@K",
        ("link prediction", "BPR", "pairwise ranking", "Recall@K", "ranking loss", "recommendation"),
        "link prediction은 아직 없는 edge가 생길 가능성을 예측하는 문제이고, 추천에서는 사용자가 좋아할 item을 상위에 놓는 문제로 나타납니다. BPR은 관측한 양성 item이 관측하지 않은 음성 item보다 높은 점수를 갖도록 pairwise로 학습합니다.",
        "사용자에게 ‘좋아할 가능성이 높은 항목이 싫어할 항목보다 위에 오게’ 순서를 맞추는 일입니다. 절대 점수보다 상대 순서가 중요합니다.",
        "그래프 추천의 loss, negative sampling, top-k 평가가 정확도와 다른 이유를 이해하게 합니다.",
        "코드에서는 user-positive-negative triplet을 만들고 score(u,i+) > score(u,i-)가 되게 합니다. Recall@K는 정답 item이 상위 K 안에 있는지 셉니다.",
        "관측되지 않은 interaction은 진짜 음성이라는 보장이 없습니다. implicit feedback의 negative 정의는 모델 가정입니다.",
    ),
    background_concept(
        "대조학습·InfoNCE·positive/negative",
        ("contrastive", "InfoNCE", "positive", "negative", "temperature", "mutual information", "dual encoder"),
        "대조학습은 같은 대상에서 만든 positive pair의 표현은 가깝게, 다른 대상의 negative pair는 멀게 만드는 학습입니다. InfoNCE는 positive 점수를 분모의 여러 후보와 비교하는 softmax형 손실이며, temperature는 점수 분포의 날카로움을 조절합니다.",
        "한 이미지의 두 crop은 같은 사람이라고 알려 주고, 다른 이미지들과는 구분하게 하는 게임입니다. 잘 학습되면 label 없이도 유용한 표현 공간이 생깁니다.",
        "SimCLR, CLIP, graph contrastive, retrieval에서 batch·queue·augmentation이 왜 중요한지 설명합니다.",
        "코드에서는 정규화한 [B, D] 표현의 similarity matrix [B, B]를 만들고 대각선 또는 짝의 위치를 정답으로 둡니다. batch가 작으면 negative 다양성이 줄 수 있습니다.",
        "negative가 항상 ‘의미적으로 틀린 예시’인 것은 아닙니다. false negative가 많으면 표현 학습이 왜곡될 수 있습니다.",
    ),
    background_concept(
        "augmentation·EMA·stop-gradient·collapse",
        ("augmentation", "EMA", "exponential moving average", "stop-gradient", "collapse", "projection MLP", "pseudo-label"),
        "augmentation은 같은 샘플의 의미는 유지하되 보이는 방식만 바꾸는 변환입니다. EMA teacher는 student 파라미터의 이동 평균을 target으로 쓰고, stop-gradient는 한 branch로 gradient가 흐르지 않게 해 target을 안정화합니다. collapse는 모든 입력이 거의 같은 표현으로 모이는 실패입니다.",
        "학생은 계속 배우고, 교사는 학생의 최근 여러 버전을 평균낸 차분한 기준을 제공합니다. 둘이 서로를 그대로 베끼기만 하면 모든 답이 같아지는 collapse가 생길 수 있습니다.",
        "BYOL·DINO·JEPA의 teacher-student 설계와 masked/self-supervised objective의 안정 장치를 읽는 데 필요합니다.",
        "코드에서는 teacher를 no_grad로 계산하고 EMA update를 optimizer.step 뒤에 합니다. augmentation이 의미를 너무 훼손하면 positive pair 가정이 깨집니다.",
        "EMA는 학습률 감소와 같은 것이 아닙니다. 파라미터의 시간 평균을 만든다는 점에서 역할이 다릅니다.",
    ),
    background_concept(
        "masked modeling·pseudo-label·zero-shot",
        ("masked", "masking", "masked LM", "masked token", "pseudo-label", "zero-shot", "linear probing", "dense probing"),
        "masked modeling은 입력 일부를 가리고 남은 문맥으로 그것을 예측하게 하는 자기지도 학습입니다. pseudo-label은 모델 또는 교사가 만든 임시 라벨이고, zero-shot은 새 task의 정답 예시 없이도 text prompt나 사전학습 지식으로 수행하는 설정입니다.",
        "문장의 빈칸이나 가려진 이미지 조각을 맞히게 하면, 정답 라벨 없이도 주변 구조를 배우게 됩니다. pseudo-label은 선생님이 만든 연습용 답안입니다.",
        "BERT·MAE·DINO 계열의 objective, foundation model의 전이 성능, 데이터 엔진을 읽는 데 필요합니다.",
        "코드에서는 mask 위치만 loss에 포함하는지, teacher label의 confidence threshold를 두는지 확인하세요. probe는 backbone을 고정하고 작은 head만 학습하는 평가입니다.",
        "pseudo-label은 정답이 아닙니다. 오류가 누적될 수 있으므로 confidence·데이터 다양성·재학습 전략이 중요합니다.",
    ),
    background_concept(
        "교차모달·CLIP·cross-attention",
        ("CLIP", "multimodal", "cross-modal", "cross-attention", "Q-Former", "text encoder", "frozen LLM", "grounding"),
        "교차모달 모델은 이미지·텍스트처럼 서로 다른 데이터 형식을 공통 표현 공간이나 attention 연결로 다룹니다. CLIP은 짝지어진 이미지와 문장을 대조학습으로 정렬하고, cross-attention은 한 모달의 query가 다른 모달의 key/value를 읽게 합니다.",
        "이미지와 문장을 같은 좌표계에 놓아 ‘이 문장이 이 이미지와 맞는가’를 비교하거나, 언어가 이미지를 필요한 만큼 참고해 답하도록 만드는 방식입니다.",
        "open-vocabulary detection, VLM, Flamingo·BLIP류의 bridge module, grounding 논문에서 모달리티가 만나는 지점을 해석하게 합니다.",
        "코드에서는 image embedding과 text embedding의 [B, D] 유사도 행렬, 또는 text query와 image token의 cross-attention을 봅니다. 어느 쪽이 query인지가 정보 흐름을 결정합니다.",
        "공통 embedding 공간이 있다고 해서 이미지와 텍스트의 모든 의미가 완벽히 정렬되는 것은 아닙니다. 데이터의 편향과 짝의 품질이 중요합니다.",
    ),
    background_concept(
        "지식 증류·teacher-student·temperature",
        ("knowledge distillation", "distillation", "teacher-student", "teacher student", "temperature", "soft targets"),
        "지식 증류는 큰 teacher 모델의 soft logit 분포를 작은 student가 모방하게 해, 정답 라벨만으로는 드러나지 않는 클래스 간 유사성까지 전달하는 방법입니다. temperature를 높이면 확률 분포가 부드러워져 작은 차이도 학습 신호가 됩니다.",
        "정답만 알려 주는 대신 숙련자가 ‘이 답도 조금 가능하지만 저 답은 거의 아니다’라는 판단의 강약까지 전수하는 방식입니다.",
        "경량 모델의 정확도 보존, logits·feature·relation distillation, 온도와 KL loss의 역할을 읽는 기본입니다.",
        "코드에서는 teacher logits를 detach하고, softmax(logits/T)와 log_softmax(student/T)로 KL을 계산합니다. 보통 T² 보정과 hard-label loss를 함께 둡니다.",
        "teacher가 틀린 편향까지 student에 전달할 수 있습니다. 증류는 teacher를 무비판적으로 복사하는 절차가 아닙니다.",
    ),
    background_concept(
        "pruning·sparsity·importance",
        ("pruning", "sparsity", "magnitude", "structured sparsity", "unstructured", "Hessian", "calibration"),
        "pruning은 영향이 작다고 판단한 weight·channel·head를 제거해 모델을 희소하게 만드는 방법입니다. unstructured sparsity는 개별 weight를, structured sparsity는 channel·block처럼 하드웨어가 활용하기 쉬운 단위를 없앱니다.",
        "짐에서 덜 중요한 물건을 빼 무게를 줄이는 일입니다. 무엇이 덜 중요한지는 weight 크기, activation, 2차 정보, calibration data로 판단할 수 있습니다.",
        "압축 논문의 정확도-속도 차이, one-shot pruning, reconstruction loss와 실제 kernel 가속의 간극을 읽는 데 필요합니다.",
        "코드에서는 mask를 곱하거나 weight를 재배열합니다. 90%가 0이어도 일반 GPU kernel이 자동으로 10배 빨라지는 것은 아닙니다.",
        "희소도와 latency는 같은 지표가 아닙니다. 구조와 라이브러리 지원이 없으면 이론적 FLOPs 감소가 실제 속도로 이어지지 않을 수 있습니다.",
    ),
    background_concept(
        "affine quantization·scale·zero-point·STE",
        ("quantization", "affine quantization", "scale", "zero-point", "fake quantization", "STE", "straight-through", "W8A8", "PTQ", "activation outlier"),
        "양자화는 float 값을 적은 비트의 정수로 근사해 메모리와 계산을 줄이는 방법입니다. affine quantization은 q≈round(x/scale)+zero-point로 실수 범위를 정수 범위에 맞추며, STE는 rounding의 미분 불가능성을 학습 때는 대략 항등 미분으로 취급합니다.",
        "긴 자를 눈금이 적은 자로 바꾸는 일입니다. scale은 눈금 간격, zero-point는 실수 0이 대응하는 정수 눈금입니다. 값 분포에 튀는 outlier가 있으면 같은 눈금으로 표현하기 어려워집니다.",
        "integer-only inference, QAT·PTQ, LLM activation 문제, 저비트 압축의 오차 원인을 설명합니다.",
        "코드에서는 clamp→round→dequantize 흐름과 per-tensor/per-channel scale을 봅니다. accumulator dtype과 clipping 범위를 함께 확인하세요.",
        "비트 수만 낮춘다고 자동으로 빨라지지 않습니다. 지원되는 kernel, memory bandwidth, dequantize 비용이 실제 성능을 결정합니다.",
    ),
    background_concept(
        "저랭크·SVD·low-rank residual",
        ("SVD", "singular value", "low-rank", "low rank", "low-rank residual", "matrix factorization", "rank"),
        "저랭크 분해는 큰 행렬을 작은 두 행렬의 곱으로 근사해 중요한 방향만 남기는 방법입니다. SVD는 행렬을 직교 기저와 singular value로 분해해, 어느 방향이 정보를 많이 담는지 보여 줍니다.",
        "해상도 높은 사진을 몇 개의 주요 색·모양 층으로 요약하는 것처럼, 행렬의 반복적·중복된 구조를 적은 축으로 표현합니다.",
        "LoRA, 모델 압축, low-rank branch, graph spectral 방법에서 계산량과 보존할 정보의 trade-off를 설명합니다.",
        "코드에서는 torch.linalg.svd 또는 저랭크 A@B를 만납니다. rank r을 줄이면 파라미터는 줄지만 근사 오차는 보통 커집니다.",
        "저랭크라는 말은 단순히 작은 행렬이라는 뜻이 아닙니다. 유효 독립 방향의 수가 작다는 선형대수 성질입니다.",
    ),
    background_concept(
        "FLOPs·latency·memory bandwidth·GPU kernel",
        ("FLOPs", "latency", "memory access", "memory bandwidth", "GPU kernel", "kernel fusion", "operator fragmentation", "hardware constraint"),
        "FLOPs는 이론적 부동소수점 연산 횟수, latency는 실제 한 요청이 끝날 때까지 걸리는 시간입니다. 메모리 bandwidth는 데이터를 읽고 쓰는 속도 한계이고, GPU kernel은 GPU에서 실행되는 연산 단위입니다.",
        "계산 횟수가 적어도 데이터를 옮기는 시간이 길거나 작은 연산을 너무 많이 호출하면 실제는 느릴 수 있습니다. 그래서 kernel fusion은 여러 연산을 한 번에 실행해 왕복 비용을 줄입니다.",
        "경량 CNN, pruning, quantization, architecture search에서 ‘작아졌다’와 ‘빨라졌다’가 왜 다른지 설명합니다.",
        "실습에서는 parameter count뿐 아니라 batch size, device 동기화, warm-up 뒤의 latency를 측정하세요. GPU 시간 측정은 비동기 실행을 고려해야 합니다.",
        "FLOPs가 절반이라고 latency가 정확히 절반이 되는 것은 아닙니다. 하드웨어·입력 크기·라이브러리 구현이 모두 관여합니다.",
    ),
    background_concept(
        "NAS·supernet·subnet·weight sharing",
        ("neural architecture search", "NAS", "supernet", "subnet", "weight sharing", "progressive shrinking", "elastic width", "elastic depth"),
        "Neural Architecture Search는 layer 수·채널·kernel·해상도 같은 구조 선택을 자동화하는 방법입니다. supernet은 여러 후보 subnet이 공통 weight를 공유하는 큰 네트워크이고, progressive shrinking은 큰 subnet부터 점차 작은 후보를 학습시키는 방식입니다.",
        "모든 집을 따로 짓지 않고, 하나의 큰 조립식 집에서 필요한 방 조합을 골라 시험하는 방식입니다. weight sharing은 탐색 비용을 줄이지만 후보 간 간섭도 만듭니다.",
        "Once-for-All처럼 다양한 기기 제약에 맞는 모델을 한 번의 학습에서 뽑아내는 논문의 핵심 구조입니다.",
        "코드에서는 width/depth/kernel choice에 따라 일부 channel과 layer만 활성화합니다. subnet을 평가할 때 BatchNorm 통계 재보정이 필요한 경우가 있습니다.",
        "supernet에서 좋은 후보가 독립적으로 처음부터 학습했을 때도 항상 최고라는 보장은 없습니다. weight sharing은 탐색 근사입니다.",
    ),
    background_concept(
        "Dueling DQN의 Q·V·A 분해",
        ("Dueling DQN", "Q/V/A", "identifiability", "mean-centering", "shared feature backbone"),
        "Dueling DQN은 행동과 무관한 상태의 좋음 V(s)와, 그 상태에서 행동별 상대적 이점 A(s,a)를 따로 예측한 뒤 Q(s,a)로 결합합니다. 보통 Q(s,a)=V(s)+A(s,a)-평균_a A(s,a)처럼 평균을 빼서 V와 A가 임의로 상쇄되는 식별 불가능성(identifiability)을 줄입니다.",
        "운전 중 ‘이 길 자체가 좋은가’와 ‘여기서 좌회전이 직진보다 나은가’를 분리해 생각하는 방식입니다. 행동 차이가 작아도 상태 가치 자체는 유용하게 배울 수 있습니다.",
        "DQN의 가치 추정 분해, shared backbone과 두 head가 왜 필요한지, mean-centering 항이 왜 수식에 들어가는지를 읽는 데 필요합니다.",
        "코드에서는 공통 feature를 만든 뒤 value head는 [B, 1], advantage head는 [B, actions]를 내고 마지막 축 평균을 뺍니다. broadcasting이 의도대로 action 축에 적용되는지 확인하세요.",
        "V와 A를 단순히 더하면 같은 Q를 만드는 조합이 너무 많습니다. mean-centering은 Q의 정의가 아니라 두 head를 안정적으로 학습시키기 위한 결합 규칙입니다.",
    ),
)


H3_PAPER_SECTION_RE = re.compile(r"^###\s+([①-⑧]\s+.+?)\s*$")
BOLD_PAPER_SECTION_RE = re.compile(r"^\s*\*\*([①-⑧]\s*.+?)\*\*\s*(.*)$")
ORDERED_PAPER_SECTION_RE = re.compile(r"^\s*([1-8])\.\s+\*\*([^*]+?)\*\*\s*(.*)$")


def clean_paper_section_title(title: str) -> str:
    cleaned = re.sub(r"^[①-⑧]\s*", "", title)
    cleaned = re.sub(r"^[1-8]\.\s*", "", cleaned)
    return cleaned.strip(" .:")


def paper_section_marker(line: str) -> tuple[str, str] | None:
    """Return a normalized numbered section title and its remaining text."""
    heading = H3_PAPER_SECTION_RE.match(line)
    if heading:
        return clean_paper_section_title(heading.group(1)), ""
    bold = BOLD_PAPER_SECTION_RE.match(line)
    if bold:
        return clean_paper_section_title(bold.group(1)), bold.group(2)
    ordered = ORDERED_PAPER_SECTION_RE.match(line)
    if ordered:
        return clean_paper_section_title(ordered.group(2)), ordered.group(3)
    return None


def paper_segment_kind(title: str) -> str:
    lowered = title.replace(" ", "")
    if "선수지식" in lowered or "기초지식" in lowered or "배경지식" in lowered:
        return "background"
    if any(keyword in lowered for keyword in ("실습", "TODO", "정답", "포트폴리오", "코드연결")):
        return "practice"
    if "자가점검" in lowered and not any(keyword in lowered for keyword in ("5분", "마무리", "정리", "한계")):
        return "check"
    return "core"


def split_paper_segments(lines: list[str]) -> list[PaperSegment]:
    """Move background, code and questions out of the main explanatory flow.

    The original notes use three authoring styles (bold labels, ordered labels,
    and H3 labels).  This splitter intentionally preserves every source line
    while changing only where it appears in the reader.
    """
    segments: list[PaperSegment] = []
    current = PaperSegment("논문 정보", "core", [])

    def flush() -> None:
        nonlocal current
        if any(line.strip() for line in current.lines):
            segments.append(current)

    for line in lines:
        marker = paper_section_marker(line)
        if marker is not None:
            flush()
            title, remainder = marker
            current = PaperSegment(
                title,
                paper_segment_kind(title),
                [remainder] if remainder else [],
                split_questions_from_lists="자가점검" in title,
            )
            continue

        if current.split_questions_from_lists and LIST_RE.match(line):
            flush()
            current = PaperSegment("이해 점검 (선택)", "check", [line])
            continue

        if "자가점검" in line:
            before, marker_text, after = line.partition("자가점검")
            if before.strip():
                current.lines.append(before.rstrip(" :.-"))
            flush()
            current = PaperSegment("이해 점검 (선택)", "check", [])
            remaining = after.lstrip(" :.-*\t")
            if remaining:
                current.lines.append(remaining)
            continue

        current.lines.append(line)

    flush()
    return segments


def normalize_background_text(value: str) -> str:
    """Normalize Korean/English prerequisite phrases for alias matching."""
    return re.sub(r"[^0-9a-z가-힣]+", "", value.casefold())


def matching_background_concepts(lines: list[str]) -> list[tuple[BackgroundConcept, tuple[str, ...]]]:
    """Return just the detailed cards relevant to a paper's own prerequisites."""
    source_text = normalize_background_text(" ".join(lines))
    matches: list[tuple[BackgroundConcept, tuple[str, ...]]] = []
    for concept in BACKGROUND_CONCEPTS:
        aliases = tuple(
            alias
            for alias in concept.aliases
            if normalize_background_text(alias) and normalize_background_text(alias) in source_text
        )
        if aliases:
            matches.append((concept, aliases))
    return matches


def render_background_concept(
    concept: BackgroundConcept,
    matched_aliases: tuple[str, ...],
    *,
    is_open: bool,
) -> str:
    """Render one expandable mini-textbook entry inside a paper's background drawer."""
    open_attr = " open" if is_open else ""
    source_words = ", ".join(escape(alias) for alias in matched_aliases[:4])
    return f'''
      <details class="background-concept"{open_attr}>
        <summary><span class="concept-name">{escape(concept.title)}</span><span class="concept-open-hint">상세 보기</span></summary>
        <div class="concept-body">
          <p class="concept-source"><b>이 논문의 원문 표현:</b> {source_words}</p>
          <dl class="concept-explanation">
            <div><dt>무엇인가</dt><dd>{escape(concept.definition)}</dd></div>
            <div><dt>직관</dt><dd>{escape(concept.intuition)}</dd></div>
            <div><dt>왜 이 논문에 필요한가</dt><dd>{escape(concept.why_it_matters)}</dd></div>
            <div><dt>코드에서 만나는 곳</dt><dd>{escape(concept.code_connection)} <a href="AI_Function_Glossary.html#overview">함수·API 사전 열기</a></dd></div>
            <div><dt>자주 헷갈리는 점</dt><dd>{escape(concept.common_confusion)}</dd></div>
          </dl>
        </div>
      </details>
    '''


def render_background_concepts(
    section: PaperSegment,
    source_dir: Path,
    source_root: Path,
) -> str:
    """Turn a terse prerequisite line into ordered, detailed concept explanations."""
    matches = matching_background_concepts(section.lines)
    raw_source = render_blocks(section.lines, source_dir, source_root)
    if not matches:
        return f'''
          <section class="knowledge-library knowledge-library-fallback">
            <p class="background-lead"><b>이 논문을 읽기 위한 연결 개념</b>을 아래 원문 표현과 함께 확인하세요. 이 항목은 아직 공통 사전에 연결되지 않아, 해당 표현이 쓰인 본문의 문장과 코드 연결을 먼저 읽도록 남겨 두었습니다.</p>
            <div class="source-prerequisite"><b>이 논문의 선수지식 표기</b>{raw_source}</div>
          </section>
        '''

    order = "".join(
        f"<li><b>{index + 1}.</b> {escape(concept.title)}</li>"
        for index, (concept, _) in enumerate(matches[:3])
    )
    cards = "".join(
        render_background_concept(concept, aliases, is_open=index < 3)
        for index, (concept, aliases) in enumerate(matches)
    )
    return f'''
      <section class="knowledge-library">
        <p class="background-lead"><b>키워드 목록이 아니라, 필요한 개념을 순서대로 풀어 둔 미니 교재입니다.</b> 아래의 처음 세 개를 먼저 읽고, 각 카드를 누르면 정의·직관·논문에서의 역할·코드 연결·주의점을 볼 수 있습니다.</p>
        <ol class="background-reading-order">{order}</ol>
        <div class="background-concept-list">{cards}</div>
        <div class="source-prerequisite"><b>이 논문의 선수지식 표기</b>{raw_source}</div>
      </section>
    '''


def render_paper_detail(lines: list[str], source_dir: Path, source_root: Path) -> str:
    """Render a paper as detailed prose first and optional learning aids second."""
    sections = split_paper_segments(lines)
    core_sections = [section for section in sections if section.kind == "core"]
    optional_sections = [section for section in sections if section.kind != "core"]

    primary_html = "".join(
        f'<section class="explanation-block"><h3>{inline_html(section.title, source_dir, source_root)}</h3>'
        f'{render_blocks(section.lines, source_dir, source_root)}</section>'
        for section in core_sections
    )

    optional_html: list[str] = []
    for section in optional_sections:
        if section.kind == "background":
            label = "상세 배경지식"
            extra = "원문의 짧은 선수지식 목록을 클릭 가능한 상세 개념 카드로 풀었습니다. 먼저 읽을 순서와 각 개념의 정의·직관·수식/shape 관점·코드 연결을 확인할 수 있습니다."
            body = render_background_concepts(section, source_dir, source_root)
        elif section.kind == "practice":
            label = "코드 · 실습 연결"
            extra = '함수가 낯설면 <a href="AI_Function_Glossary.html">함수·API 한국어 사전</a>에서 입력·반환·주의점을 먼저 확인할 수 있습니다.'
            body = render_blocks(section.lines, source_dir, source_root)
        else:
            label = "이해 점검"
            extra = "정답을 요구하는 화면이 아닙니다. 읽은 뒤 생각을 정리하고 싶을 때만 사용하세요."
            body = render_blocks(section.lines, source_dir, source_root)
        optional_html.append(
            f'<details class="optional-block optional-{section.kind}">'
            f'<summary><span>{label}</span><b>{inline_html(section.title, source_dir, source_root)}</b></summary>'
            f'<div class="optional-body"><p class="optional-intro">{extra}</p>'
            f'{body}</div></details>'
        )

    if not primary_html:
        primary_html = f'<section class="explanation-block">{render_blocks(lines, source_dir, source_root)}</section>'
    return f'<div class="detailed-explanation">{primary_html}</div><div class="optional-aids">{"".join(optional_html)}</div>'


def render_field(field: FieldSpec, notes_dir: Path, source_root: Path) -> tuple[str, int]:
    source_path = notes_dir / field.filename
    lines = source_path.read_text(encoding="utf-8").splitlines()
    intro, sections = split_h2_sections(lines)
    intro_html = render_blocks(intro, source_path.parent, source_root)
    paper_sections: list[tuple[str, str, str]] = []
    supplements: list[str] = []

    for heading, body in sections:
        numbered = NUMBERED_PAPER_RE.match(heading)
        if numbered:
            number, title = numbered.groups()
            paper_id = f"paper-{field.slug}-{number}"
            paper_sections.append((paper_id, number, title))
            rendered_body = render_paper_detail(body, source_path.parent, source_root)
            supplements.append(
                f'<details class="paper" id="{paper_id}" data-field="{field.slug}">'
                "<summary>"
                f'<span class="paper-number">{escape(number)}</span>'
                f'<span class="paper-title">{inline_html(title, source_path.parent, source_root)}</span>'
                '<span class="open-hint">열기</span>'
                "</summary>"
                f'<div class="paper-body">{rendered_body}</div>'
                "</details>"
            )
        else:
            rendered_body = render_blocks(body, source_path.parent, source_root)
            supplements.append(
                f'<section class="supplement"><h2>{inline_html(heading, source_path.parent, source_root)}</h2>{rendered_body}</section>'
            )

    toc_items = "".join(
        f'<a href="#{paper_id}" data-open-paper="{paper_id}"><b>{escape(number)}</b>{inline_html(title, source_path.parent, source_root)}</a>'
        for paper_id, number, title in paper_sections
    )
    panel = f'''
      <section id="field-{field.slug}" class="field-panel" data-field-panel="{field.slug}" hidden>
        <div class="field-heading">
          <div>
            <p class="eyebrow">핵심 논문 10편 · 한국어 독해 노트</p>
            <h1>{escape(field.label)}</h1>
            <p>{escape(field.route)} · {escape(field.short_label)}</p>
          </div>
          <button class="quiet-button" data-show-overview>분야 선택으로 돌아가기</button>
        </div>
        <div class="reader-callout"><b>상세 해설 우선</b><span>논문의 문제, 방법, 수식 직관, 원문 위치와 한계를 본문에서 먼저 읽습니다. 배경 지식은 아래 선택 영역에서 키워드 목록이 아니라 정의·직관·코드 연결을 갖춘 상세 카드로 펼쳐집니다.</span></div>
        <div class="field-intro">{intro_html}</div>
        <nav class="paper-index" aria-label="{escape(field.label)} 논문 목차">{toc_items}</nav>
        <div class="paper-list">{''.join(supplements)}</div>
      </section>
    '''
    return panel, len(paper_sections)


def render_support_document(path: Path, source_root: Path, section_id: str, title: str, intro: str) -> str:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("# ")]
    body = render_blocks(lines, path.parent, source_root)
    return f'''
      <section id="{section_id}" class="support-panel" hidden>
        <div class="field-heading">
          <div><p class="eyebrow">보조 자료</p><h1>{escape(title)}</h1><p>{escape(intro)}</p></div>
          <button class="quiet-button" data-show-overview>분야 선택으로 돌아가기</button>
        </div>
        <div class="support-body">{body}</div>
      </section>
    '''


def build_html(source_root: Path) -> str:
    notes_dir = source_root / "docs" / "paper_reading_notes"
    required = [notes_dir / field.filename for field in FIELDS]
    required.extend([notes_dir / "RECENT_TOP_TIER.md", notes_dir / "AI_ASSISTED_NOTICE.md"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Korean paper-note source is missing: " + ", ".join(missing))

    fields_html: list[str] = []
    total_papers = 0
    for field in FIELDS:
        panel, count = render_field(field, notes_dir, source_root)
        fields_html.append(panel)
        total_papers += count
    if total_papers != 70:
        raise ValueError(f"Expected 70 numbered core papers, found {total_papers}.")

    field_cards = "".join(
        f'''
        <button class="field-card" data-show-field="{field.slug}">
          <span class="field-count">10편</span>
          <strong>{escape(field.label)}</strong>
          <span>{escape(field.short_label)}</span>
        </button>'''
        for field in FIELDS
    )
    radar_html = render_support_document(
        notes_dir / "RECENT_TOP_TIER.md",
        source_root,
        "radar",
        "2023–2025 최신 논문 레이더",
        "핵심 70편 다음에 연결할 후속 논문 지도입니다.",
    )
    notice_html = render_support_document(
        notes_dir / "AI_ASSISTED_NOTICE.md",
        source_root,
        "notice",
        "이 자료를 읽고 인용하는 방법",
        "한국어 독해 노트의 작성 범위와 정확성 한계를 확인하세요.",
    )

    return f'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>AI 핵심 논문 한국어 요약 리더</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5e6b7f;
      --line: #dbe3ee;
      --surface: #ffffff;
      --page: #f4f7fb;
      --navy: #172b4d;
      --blue: #2f6feb;
      --blue-soft: #eaf2ff;
      --teal: #0f766e;
      --amber: #a16207;
      --shadow: 0 14px 34px rgba(21, 39, 70, .10);
      --reading-size: 17px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--page); color: var(--ink); font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif; line-height: 1.78; font-size: var(--reading-size); }}
    button, input {{ font: inherit; }}
    button {{ cursor: pointer; }}
    a {{ color: #1558c0; text-underline-offset: 3px; }}
    .shell {{ max-width: 1230px; margin: 0 auto; padding: 0 24px 72px; }}
    .hero {{ margin: 26px 0 18px; padding: 42px 46px; border-radius: 24px; color: white; background: radial-gradient(circle at 82% 18%, rgba(94, 180, 255, .42), transparent 27%), linear-gradient(130deg, #132745, #254d84); box-shadow: var(--shadow); }}
    .eyebrow {{ margin: 0 0 9px; font-size: 13px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #6ea8ff; }}
    .hero h1 {{ max-width: 780px; margin: 0; font-size: clamp(30px, 5vw, 47px); line-height: 1.24; letter-spacing: -.04em; }}
    .hero p {{ max-width: 790px; margin: 16px 0 0; color: #dce9ff; }}
    .hero-stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 25px; }}
    .stat {{ min-width: 145px; padding: 10px 14px; border: 1px solid rgba(255,255,255,.24); border-radius: 12px; background: rgba(255,255,255,.10); }}
    .stat strong {{ display: block; font-size: 22px; line-height: 1.2; }}
    .stat span {{ font-size: 13px; color: #dce9ff; }}
    .toolbar {{ position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 12px; padding: 13px 0; background: linear-gradient(var(--page) 78%, rgba(244,247,251,0)); }}
    .search-wrap {{ position: relative; flex: 1; }}
    .search-wrap input {{ width: 100%; padding: 12px 15px 12px 43px; color: var(--ink); border: 1px solid #cbd7e6; border-radius: 12px; background: white; box-shadow: 0 2px 7px rgba(25, 47, 80, .05); }}
    .search-wrap::before {{ content: "⌕"; position: absolute; top: 7px; left: 15px; color: #4d6382; font-size: 25px; line-height: 1; }}
    .tool-button, .quiet-button {{ min-height: 43px; padding: 8px 13px; border: 1px solid #cbd7e6; border-radius: 10px; color: #284465; background: white; font-size: 14px; font-weight: 700; }}
    .tool-button:hover, .quiet-button:hover {{ border-color: #6b9af1; background: #f3f7ff; }}
    .search-results {{ display: none; margin: 0 0 22px; padding: 15px; border: 1px solid #bad0f4; border-radius: 15px; background: #f8fbff; }}
    .search-results.is-visible {{ display: block; }}
    .search-results p {{ margin: 0 0 8px; color: var(--muted); font-size: 14px; }}
    .result-list {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .result-list button {{ padding: 7px 10px; border: 1px solid #c9dbfa; border-radius: 9px; color: #164c9d; background: white; font-size: 13px; text-align: left; }}
    .result-list button:hover {{ background: var(--blue-soft); }}
    .overview, .field-panel, .support-panel {{ padding: 30px; border: 1px solid var(--line); border-radius: 20px; background: var(--surface); box-shadow: 0 6px 21px rgba(24, 48, 85, .06); }}
    .overview h2 {{ margin: 0; font-size: 27px; letter-spacing: -.03em; }}
    .overview > p {{ margin: 9px 0 21px; color: var(--muted); }}
    .reading-route {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 20px 0 26px; }}
    .route-card {{ padding: 18px; border-radius: 13px; background: #f4f8fe; }}
    .route-card b {{ display: block; margin-bottom: 4px; color: #224a82; }}
    .route-card span {{ display: block; color: #526176; font-size: 14px; line-height: 1.6; }}
    .field-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 13px; }}
    .field-card {{ position: relative; min-height: 130px; padding: 20px 20px 18px; overflow: hidden; border: 1px solid #d8e2ef; border-radius: 15px; color: var(--ink); background: white; text-align: left; transition: .16s ease; }}
    .field-card::after {{ content: "→"; position: absolute; right: 18px; bottom: 12px; color: #4a7bc8; font-size: 22px; }}
    .field-card:hover {{ transform: translateY(-2px); border-color: #79a4ea; box-shadow: 0 8px 19px rgba(43, 90, 160, .12); }}
    .field-card strong {{ display: block; margin: 6px 0 3px; font-size: 20px; letter-spacing: -.02em; }}
    .field-card span:not(.field-count) {{ color: var(--muted); font-size: 14px; }}
    .field-count {{ display: inline-block; padding: 2px 8px; border-radius: 999px; color: #1e5cb8; background: var(--blue-soft); font-size: 12px; font-weight: 800; }}
    .secondary-links {{ display: flex; flex-wrap: wrap; gap: 9px; margin-top: 24px; }}
    .secondary-links button, .secondary-links a {{ padding: 10px 13px; border: 1px solid #d4dfed; border-radius: 10px; color: #34526f; background: #fbfdff; font-size: 14px; font-weight: 700; text-decoration: none; }}
    .secondary-links button:hover, .secondary-links a:hover {{ background: #f1f6ff; }}
    .field-heading {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; padding-bottom: 19px; border-bottom: 1px solid var(--line); }}
    .field-heading h1 {{ margin: 0; font-size: 32px; letter-spacing: -.035em; line-height: 1.28; }}
    .field-heading p:not(.eyebrow) {{ margin: 7px 0 0; color: var(--muted); }}
    .reader-callout {{ display: flex; gap: 12px; align-items: flex-start; margin: 22px 0; padding: 15px 17px; border-left: 4px solid var(--teal); border-radius: 8px 13px 13px 8px; background: #eefaf7; }}
    .reader-callout b {{ flex: 0 0 auto; color: #075f57; }}
    .reader-callout span {{ color: #385c59; font-size: 15px; line-height: 1.65; }}
    .field-intro {{ color: #445267; font-size: 15px; }}
    .field-intro > :first-child {{ margin-top: 0; }}
    .paper-index {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 25px 0; }}
    .paper-index a {{ display: flex; gap: 8px; align-items: baseline; min-width: 0; padding: 9px 10px; border: 1px solid #dde6f1; border-radius: 9px; color: #294669; background: #fbfdff; text-decoration: none; font-size: 14px; line-height: 1.45; }}
    .paper-index a:hover {{ border-color: #89b0ef; background: #f0f6ff; }}
    .paper-index b {{ color: #2e6dd5; font-size: 12px; }}
    .paper-list {{ display: grid; gap: 12px; }}
    details.paper {{ overflow: clip; border: 1px solid #d9e3ef; border-radius: 13px; background: white; }}
    details.paper[open] {{ border-color: #a8c4ee; box-shadow: 0 7px 20px rgba(26, 74, 140, .08); }}
    details.paper summary {{ display: flex; align-items: center; gap: 12px; padding: 16px 17px; list-style: none; font-weight: 800; cursor: pointer; }}
    details.paper summary::-webkit-details-marker {{ display: none; }}
    details.paper summary::before {{ content: "+"; flex: 0 0 22px; color: #2464c4; font-size: 21px; line-height: 1; text-align: center; }}
    details.paper[open] summary::before {{ content: "−"; }}
    .paper-number {{ flex: 0 0 auto; padding: 3px 7px; border-radius: 7px; color: #1754aa; background: #eaf2ff; font-size: 12px; }}
    .paper-title {{ min-width: 0; color: #1d2b41; line-height: 1.5; }}
    .open-hint {{ margin-left: auto; color: #708199; font-size: 12px; font-weight: 600; }}
    .paper-body {{ padding: 2px 22px 25px 52px; color: #313e52; }}
    .paper-body h3 {{ margin: 25px 0 8px; color: #1c4f91; font-size: 17px; }}
    .paper-body h4 {{ margin: 20px 0 7px; color: #334e72; font-size: 16px; }}
    .paper-body p, .support-body p {{ margin: 10px 0 15px; }}
    .paper-body ul, .paper-body ol, .support-body ul, .support-body ol {{ padding-left: 24px; }}
    .paper-body li, .support-body li {{ margin: 6px 0; }}
    .paper-body code, .support-body code {{ padding: 1px 5px; border-radius: 4px; color: #713c0e; background: #fff4e6; font-family: Consolas, "Cascadia Code", monospace; font-size: .9em; }}
    .math {{ padding: 0 3px; color: #394c88; font-family: Cambria Math, Cambria, serif; }}
    .display-math {{ overflow-x: auto; margin: 15px 0; padding: 13px 15px; border-left: 3px solid #7da6e7; border-radius: 8px; color: #273b78; background: #f3f7ff; font-family: Cambria Math, Cambria, serif; line-height: 1.65; }}
    .detailed-explanation {{ display: grid; gap: 13px; }}
    .explanation-block {{ padding: 17px 18px; border: 1px solid #e0e8f2; border-radius: 12px; background: #fcfdff; }}
    .explanation-block:first-child {{ border-color: #bcd3f7; background: #f6faff; }}
    .explanation-block h3 {{ margin: 0 0 10px; padding-bottom: 8px; border-bottom: 1px solid #e1e9f3; color: #164e96; font-size: 17px; }}
    .explanation-block p:last-child, .explanation-block ul:last-child, .explanation-block ol:last-child {{ margin-bottom: 0; }}
    .optional-aids {{ display: grid; gap: 8px; margin-top: 18px; }}
    .optional-block {{ overflow: clip; border: 1px solid #dce5ef; border-radius: 11px; background: white; }}
    .optional-block summary {{ display: flex; gap: 10px; align-items: center; padding: 12px 14px; list-style: none; color: #354a67; cursor: pointer; }}
    .optional-block summary::-webkit-details-marker {{ display: none; }}
    .optional-block summary::before {{ content: "+"; color: #2f6feb; font-size: 19px; font-weight: 700; }}
    .optional-block[open] summary::before {{ content: "−"; }}
    .optional-block summary span {{ flex: 0 0 auto; padding: 2px 7px; border-radius: 999px; color: #245eaf; background: #eaf2ff; font-size: 12px; font-weight: 800; }}
    .optional-block summary b {{ min-width: 0; font-size: 14px; }}
    .optional-body {{ padding: 0 16px 16px 43px; color: #3c4d64; }}
    .optional-intro {{ margin-top: 0 !important; padding: 10px 12px; border-radius: 8px; color: #496070; background: #f6f9fc; font-size: 14px; }}
    .optional-background {{ border-color: #c7dfd8; }}
    .optional-background summary span {{ color: #087367; background: #e6f7f2; }}
    .knowledge-library {{ margin-top: 14px; }}
    .background-lead {{ margin: 0 0 12px !important; padding: 13px 14px; border-left: 3px solid #159886; border-radius: 9px; color: #235d56; background: #effaf7; font-size: 14px; line-height: 1.72; }}
    .background-reading-order {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; padding: 0; list-style: none; }}
    .background-reading-order li {{ padding: 6px 9px; border: 1px solid #d5e9e3; border-radius: 999px; color: #28635c; background: #fbfffe; font-size: 13px; }}
    .background-reading-order b {{ color: #087367; }}
    .background-concept-list {{ display: grid; gap: 9px; }}
    .background-concept {{ overflow: clip; border: 1px solid #cde3dc; border-radius: 10px; background: #fff; }}
    .background-concept[open] {{ border-color: #79bcae; box-shadow: 0 4px 13px rgba(11, 104, 90, .08); }}
    .background-concept summary {{ display: flex; align-items: center; gap: 10px; padding: 12px 13px; list-style: none; color: #174e48; cursor: pointer; }}
    .background-concept summary::-webkit-details-marker {{ display: none; }}
    .background-concept summary::before {{ content: "+"; color: #087367; font-size: 18px; font-weight: 800; line-height: 1; }}
    .background-concept[open] summary::before {{ content: "−"; }}
    .concept-name {{ font-weight: 800; }}
    .concept-open-hint {{ margin-left: auto; color: #66837f; font-size: 12px; font-weight: 600; }}
    .concept-body {{ padding: 0 13px 14px 39px; }}
    .concept-source {{ margin: 0 0 11px !important; color: #4c706b; font-size: 12px; line-height: 1.6; }}
    .concept-explanation {{ display: grid; gap: 8px; margin: 0; }}
    .concept-explanation > div {{ padding: 10px 11px; border-radius: 8px; background: #f7fbfa; }}
    .concept-explanation dt {{ margin: 0 0 3px; color: #087367; font-size: 12px; font-weight: 800; }}
    .concept-explanation dd {{ margin: 0; color: #385852; font-size: 14px; line-height: 1.7; }}
    .concept-explanation a {{ color: #1267be; font-weight: 700; text-decoration: none; white-space: nowrap; }}
    .concept-explanation a:hover {{ text-decoration: underline; }}
    .source-prerequisite {{ margin-top: 13px; padding: 11px 13px; border: 1px dashed #c7dcd7; border-radius: 9px; color: #59716d; background: #fbfefd; font-size: 13px; }}
    .source-prerequisite > b {{ display: block; margin-bottom: 6px; color: #486560; font-size: 12px; }}
    .source-prerequisite p {{ margin: 0 !important; line-height: 1.6; }}
    .optional-practice {{ border-color: #d8d0f2; }}
    .optional-practice summary span {{ color: #6244ab; background: #f1edff; }}
    .optional-check {{ border-color: #f0ddb5; }}
    .optional-check summary span {{ color: #8a5a06; background: #fff5df; }}
    .source-note {{ margin: 16px 0; padding: 13px 16px; border-radius: 9px; color: #5c512d; background: #fff9e6; font-size: 14px; line-height: 1.7; }}
    pre {{ overflow: auto; padding: 15px; border-radius: 10px; color: #dce8ff; background: #16253b; font-size: 13px; line-height: 1.6; }}
    pre code {{ padding: 0; color: inherit; background: transparent; }}
    .supplement {{ margin-top: 26px; padding: 22px; border-top: 3px solid #a8c4ee; border-radius: 12px; background: #f9fbff; }}
    .supplement h2 {{ margin: 0 0 14px; color: #1e4f91; font-size: 22px; }}
    .support-body {{ margin-top: 23px; color: #344258; }}
    .support-body h2 {{ margin: 30px 0 10px; color: #1f4f90; font-size: 22px; }}
    .support-body h3 {{ margin: 21px 0 8px; color: #34587e; font-size: 18px; }}
    .table-wrap {{ overflow-x: auto; margin: 16px 0; border: 1px solid #d9e4f0; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; line-height: 1.55; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e9f2; vertical-align: top; text-align: left; }}
    th {{ color: #24486d; background: #eff5fc; }}
    tr:last-child td {{ border-bottom: 0; }}
    hr {{ margin: 24px 0; border: 0; border-top: 1px solid #dce5f0; }}
    [hidden] {{ display: none !important; }}
    @media (max-width: 760px) {{
      .shell {{ padding: 0 13px 42px; }}
      .hero {{ margin-top: 13px; padding: 30px 23px; border-radius: 18px; }}
      .toolbar {{ flex-wrap: wrap; }}
      .search-wrap {{ flex-basis: 100%; }}
      .reading-route, .field-grid, .paper-index {{ grid-template-columns: 1fr; }}
      .overview, .field-panel, .support-panel {{ padding: 22px 17px; border-radius: 16px; }}
      .field-heading {{ display: block; }}
      .quiet-button {{ margin-top: 14px; }}
      .paper-body {{ padding: 1px 16px 20px; }}
      .explanation-block {{ padding: 15px; }}
      .optional-body {{ padding: 0 13px 15px; }}
      .concept-body {{ padding: 0 11px 13px; }}
      .background-reading-order {{ display: grid; grid-template-columns: 1fr; }}
      details.paper summary {{ padding: 13px; gap: 8px; }}
      .open-hint {{ display: none; }}
    }}
    @media print {{
      body {{ background: white; font-size: 12pt; }}
      .toolbar, .search-results, .quiet-button, .secondary-links, .paper-index {{ display: none !important; }}
      .hero {{ margin: 0 0 12px; padding: 20px; border-radius: 0; box-shadow: none; }}
      .overview, .field-panel, .support-panel {{ border: 0; box-shadow: none; padding: 0; }}
      details.paper {{ break-inside: avoid; }}
      details.paper:not([open]) .paper-body {{ display: block; }}
      details.paper:not([open]) summary::before {{ content: ""; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">offline reader · 상세 한국어 해설</p>
      <h1>AI 핵심 논문을<br>자세히 읽는 한국어 해설</h1>
      <p>원문을 문장 단위로 옮긴 번역본이 아니라, 논문의 문제·방법·수식 직관·의미·한계를 독자적으로 자세히 풀어 쓴 자료입니다. 실습과 질문은 본문을 방해하지 않도록 선택 영역으로 분리했습니다.</p>
      <div class="hero-stats">
        <div class="stat"><strong>{total_papers}편</strong><span>상세 한국어 해설</span></div>
        <div class="stat"><strong>7개 분야</strong><span>상세 배경 카드 포함</span></div>
        <div class="stat"><strong>37편+</strong><span>최신 후속 논문 레이더</span></div>
      </div>
    </header>

    <div class="toolbar" aria-label="논문 요약 도구">
      <label class="search-wrap"><input id="paper-search" type="search" placeholder="논문 제목·키워드로 찾기 (예: Transformer, RAG, 강화학습)" autocomplete="off"></label>
      <button id="font-down" class="tool-button" type="button" aria-label="글자 작게">가−</button>
      <button id="font-up" class="tool-button" type="button" aria-label="글자 크게">가＋</button>
      <button id="print-reader" class="tool-button" type="button">인쇄</button>
    </div>
    <section id="search-results" class="search-results" aria-live="polite"></section>

    <section id="overview" class="overview">
      <h2>어떤 논문을 자세히 읽을까요?</h2>
      <p>각 논문을 열면 <b>논문 정보 → 문제와 배경 → 핵심 방법 → 수식·알고리즘 직관 → 원문에서 확인할 위치 → 한계</b>가 기본 본문으로 나옵니다. 필요한 순간에만 배경 지식과 코드 설명을 펼치면 됩니다.</p>
      <div class="reading-route">
        <div class="route-card"><b>본문은 해설 중심</b><span>문제와 방법의 흐름을 먼저 읽습니다. 자가점검 문제는 처음에 보이지 않습니다.</span></div>
        <div class="route-card"><b>배경 지식은 필요할 때</b><span>선수지식은 개념별 상세 카드에서 정의·직관·논문 속 역할·코드 연결까지 확인합니다.</span></div>
        <div class="route-card"><b>코드 함수도 오프라인으로</b><span>실습과 연결된 API는 한국어 함수 사전에서 입력·반환·주의점을 바로 확인합니다.</span></div>
      </div>
      <div class="field-grid">{field_cards}</div>
      <div class="secondary-links">
        <a href="AI_Function_Glossary.html">함수·API 한국어 사전</a>
        <button type="button" data-show-support="radar">최신 논문 레이더 보기</button>
        <button type="button" data-show-support="notice">작성 원칙과 인용 안내</button>
      </div>
    </section>

    {''.join(fields_html)}
    {radar_html}
    {notice_html}
  </main>
  <script>
    (() => {{
      const overview = document.getElementById('overview');
      const fieldPanels = [...document.querySelectorAll('[data-field-panel]')];
      const supportPanels = [...document.querySelectorAll('.support-panel')];
      const allPanels = [...fieldPanels, ...supportPanels];
      const search = document.getElementById('paper-search');
      const results = document.getElementById('search-results');
      const root = document.documentElement;
      let fontSize = 17;

      function hideAll() {{
        overview.hidden = true;
        allPanels.forEach(panel => panel.hidden = true);
      }}
      function showOverview(updateHash = true) {{
        allPanels.forEach(panel => panel.hidden = true);
        overview.hidden = false;
        if (updateHash) history.replaceState(null, '', location.pathname + location.search);
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
      function showPanel(id, updateHash = true) {{
        const panel = document.getElementById(id.startsWith('field-') ? id : `field-${{id}}`) || document.getElementById(id);
        if (!panel) {{ showOverview(updateHash); return; }}
        hideAll();
        panel.hidden = false;
        if (updateHash) history.replaceState(null, '', `#${{panel.id.replace('field-', '')}}`);
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
      function openPaper(paperId) {{
        const paper = document.getElementById(paperId);
        if (!paper) return;
        showPanel(paper.dataset.field);
        paper.open = true;
        setTimeout(() => paper.scrollIntoView({{ behavior: 'smooth', block: 'start' }}), 80);
      }}
      function applyHash() {{
        const hash = decodeURIComponent(location.hash.replace(/^#/, ''));
        if (!hash) {{ showOverview(false); return; }}
        const fieldPanel = document.getElementById(`field-${{hash}}`);
        const supportPanel = supportPanels.find(panel => panel.id === hash);
        if (fieldPanel || supportPanel) {{ showPanel(hash, false); return; }}
        const paper = document.getElementById(hash);
        if (paper) openPaper(hash);
        else showOverview(false);
      }}

      document.querySelectorAll('[data-show-field]').forEach(button => button.addEventListener('click', () => showPanel(button.dataset.showField)));
      document.querySelectorAll('[data-show-overview]').forEach(button => button.addEventListener('click', () => showOverview()));
      document.querySelectorAll('[data-show-support]').forEach(button => button.addEventListener('click', () => showPanel(button.dataset.showSupport)));
      document.querySelectorAll('[data-open-paper]').forEach(link => link.addEventListener('click', event => {{ event.preventDefault(); openPaper(link.dataset.openPaper); }}));
      window.addEventListener('hashchange', applyHash);

      search.addEventListener('input', () => {{
        const query = search.value.trim().toLocaleLowerCase();
        if (!query) {{ results.classList.remove('is-visible'); results.innerHTML = ''; return; }}
        const matches = [...document.querySelectorAll('details.paper')]
          .filter(paper => paper.textContent.toLocaleLowerCase().includes(query))
          .slice(0, 18);
        const label = matches.length ? `검색 결과 ${{matches.length}}개${{matches.length === 18 ? ' 이상' : ''}}` : '일치하는 논문을 찾지 못했습니다.';
        const buttons = matches.map(paper => {{
          const title = paper.querySelector('.paper-title').textContent.trim();
          const field = paper.closest('[data-field-panel]').querySelector('h1').textContent.trim();
          return `<button type="button" data-result="${{paper.id}}">${{field}} · ${{title}}</button>`;
        }}).join('');
        results.innerHTML = `<p>${{label}}</p><div class="result-list">${{buttons}}</div>`;
        results.classList.add('is-visible');
        results.querySelectorAll('[data-result]').forEach(button => button.addEventListener('click', () => {{
          search.value = ''; results.classList.remove('is-visible'); results.innerHTML = ''; openPaper(button.dataset.result);
        }}));
      }});
      document.getElementById('font-down').addEventListener('click', () => {{ fontSize = Math.max(14, fontSize - 1); root.style.setProperty('--reading-size', `${{fontSize}}px`); }});
      document.getElementById('font-up').addEventListener('click', () => {{ fontSize = Math.min(23, fontSize + 1); root.style.setProperty('--reading-size', `${{fontSize}}px`); }});
      document.getElementById('print-reader').addEventListener('click', () => window.print());
      applyHash();
    }})();
  </script>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    source_root = (args.source_root or repository_root()).resolve()
    output = (args.output or source_root / "AI_Korean_Paper_Notes.html").resolve()
    html_text = build_html(source_root)
    html_text = "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(f"WROTE {output}")
    print("CORE_PAPERS=70")


if __name__ == "__main__":
    main()
