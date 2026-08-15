"""Create a compact printable learning guide for the AI coding package."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AI_Coding_Practice_Guide.pdf"
FONT_REGULAR = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")

NAVY = colors.HexColor("#163A5F")
BLUE = colors.HexColor("#216BA5")
TEAL = colors.HexColor("#008B82")
AMBER = colors.HexColor("#D78B0B")
INK = colors.HexColor("#172B3A")
MUTED = colors.HexColor("#536273")
LINE = colors.HexColor("#D8E0E8")
PALE_BLUE = colors.HexColor("#EAF3FB")
PALE_TEAL = colors.HexColor("#E5F7F5")
PALE_AMBER = colors.HexColor("#FFF5DD")
PALE_GRAY = colors.HexColor("#F5F7FA")


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "cover_kicker",
            parent=base["Normal"],
            fontName="MalgunBold",
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#BFE7FF"),
            spaceAfter=12,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="MalgunBold",
            fontSize=29,
            leading=39,
            textColor=colors.white,
            spaceAfter=14,
            wordWrap="CJK",
        ),
        "cover_body": ParagraphStyle(
            "cover_body",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=13,
            leading=21,
            textColor=colors.HexColor("#E8F3FC"),
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="MalgunBold",
            fontSize=19,
            leading=27,
            textColor=NAVY,
            spaceAfter=11,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="MalgunBold",
            fontSize=12.5,
            leading=19,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=10,
            leading=16,
            textColor=INK,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "card_title": ParagraphStyle(
            "card_title",
            parent=base["Heading3"],
            fontName="MalgunBold",
            fontSize=10.5,
            leading=16,
            textColor=NAVY,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "card_body": ParagraphStyle(
            "card_body",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8.5,
            leading=13,
            textColor=INK,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "table_head",
            parent=base["BodyText"],
            fontName="MalgunBold",
            fontSize=8.4,
            leading=12,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8.2,
            leading=12.5,
            textColor=INK,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base["BodyText"],
            fontName="MalgunBold",
            fontSize=11,
            leading=18,
            textColor=TEAL,
            leftIndent=8,
            rightIndent=8,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8,
            leading=12,
            textColor=colors.HexColor("#1F3248"),
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def box(title: str, text: str, style: dict[str, ParagraphStyle], color: colors.Color) -> Table:
    table = Table(
        [[[
            para(title, style["card_title"]),
            para(text, style["card_body"]),
        ]]],
        colWidths=[54 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def grid(cards: list[Table]) -> Table:
    rows = []
    for index in range(0, len(cards), 3):
        row = cards[index : index + 3]
        row.extend([""] * (3 - len(row)))
        rows.append(row)
    table = Table(rows, colWidths=[54 * mm, 54 * mm, 54 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    return table


def footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Malgun", 8)
    canvas.drawString(18 * mm, 8.5 * mm, "AI 코딩 실습 센터 - 읽기 쉬운 학습 가이드")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, str(doc.page))
    canvas.restoreState()


def build() -> None:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("Windows 맑은 고딕 폰트를 찾지 못했습니다.")
    pdfmetrics.registerFont(TTFont("Malgun", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MalgunBold", str(FONT_BOLD)))
    style = make_styles()
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="AI 코딩 실습 센터 - 읽기 쉬운 학습 가이드",
    )
    story: list[object] = []

    cover = Table(
        [[[
            para("AI CODING PRACTICE CENTER", style["cover_kicker"]),
            para("읽고, 직접 치고,<br/>실행하며 배우는 AI 코딩", style["cover_title"]),
            para(
                "Markdown 문서를 찾아다니지 않아도 됩니다. 이 가이드는 실습 시작점, "
                "학습 순서, 프로젝트 따라치기 루프를 한 눈에 보여 줍니다.",
                style["cover_body"],
            ),
        ]]],
        colWidths=[174 * mm],
        rowHeights=[176 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 18 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 18 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 33 * mm),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend(
        [
            cover,
            Spacer(1, 12 * mm),
            para(
                "핵심 원칙: 정답을 먼저 읽지 말고, 작은 TODO를 직접 타이핑한 뒤 "
                "실행 결과와 오류를 근거로 다음 코드를 결정합니다.",
                style["quote"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            para("1. 처음에는 세 번만 클릭하세요", style["h1"]),
            para(
                "바탕화면의 AI 코딩 실습 센터를 열어 환경 상태를 확인하세요. "
                "처음에는 설치가 필요하고, 설치가 끝나면 기본 실습 00부터 시작하면 됩니다.",
                style["body"],
            ),
            grid(
                [
                    box(
                        "01 - 설치 또는 복구",
                        "환경 관리 탭에서 실행합니다. Python 3.12, 가상환경, JupyterLab과 실습 패키지를 준비합니다.",
                        style,
                        PALE_BLUE,
                    ),
                    box(
                        "02 - 전체 환경 점검",
                        "테스트, 노트북 구조, 실행기를 확인합니다. 설치 직후 한 번 실행하면 기준점이 생깁니다.",
                        style,
                        PALE_TEAL,
                    ),
                    box(
                        "03 - 기본 실습 00",
                        "JupyterLab과 프로젝트 구조를 먼저 익힙니다. 이후 01부터 순서대로 확장합니다.",
                        style,
                        PALE_AMBER,
                    ),
                ]
            ),
            Spacer(1, 8 * mm),
            para("세 가지 학습 경로", style["h2"]),
        ]
    )
    track_data = [
        [
            para("경로", style["table_head"]),
            para("언제 고를까", style["table_head"]),
            para("시작점", style["table_head"]),
            para("얻는 감각", style["table_head"]),
        ],
        [
            para("기본 AI 커리큘럼", style["table_cell"]),
            para("Python부터 PyTorch, RAG까지 순서를 만들고 싶을 때", style["table_cell"]),
            para("00 - 환경과 JupyterLab", style["table_cell"]),
            para("데이터, 텐서, 학습 루프, 검색을 연결하는 감각", style["table_cell"]),
        ],
        [
            para("대표 논문 20편", style["table_cell"]),
            para("핵심 아이디어의 변천을 빠르게 훑고 싶을 때", style["table_cell"]),
            para("00 - LeNet-5", style["table_cell"]),
            para("논문 주장과 코드 구조를 대응시키는 감각", style["table_cell"]),
        ],
        [
            para("분야별 논문 70편", style["table_cell"]),
            para("한 분야를 깊게 파고들고 싶을 때", style["table_cell"]),
            para("관심 분야 00", style["table_cell"]),
            para("반복되는 구현 패턴과 실험 설계 감각", style["table_cell"]),
        ],
    ]
    track_table = Table(track_data, colWidths=[34 * mm, 57 * mm, 38 * mm, 45 * mm])
    track_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([track_table, PageBreak()])

    story.extend(
        [
            para("2. 매일의 직접 타이핑 루프", style["h1"]),
            para(
                "전체 정답을 복사하면 속도는 빨라져도 다음 문제에서 다시 막힙니다. "
                "작고 확실한 단위를 직접 쓰는 편이 더 오래 남습니다.",
                style["body"],
            ),
            grid(
                [
                    box("1 - 문제를 한 문장으로", "TODO 앞 설명을 읽고 입력, 출력, 실패 조건을 한 줄로 적습니다.", style, PALE_BLUE),
                    box("2 - 10-20줄 직접 입력", "함수 하나 또는 작은 클래스를 직접 타이핑합니다. 첫 버전은 완벽하지 않아도 됩니다.", style, PALE_TEAL),
                    box("3 - 즉시 실행", "starter.py 또는 노트북 셀을 실행해 실제 오류와 shape, metric을 확인합니다.", style, PALE_AMBER),
                    box("4 - 실패를 한 줄로 기록", "어떤 입력에서 왜 실패했는지 적고, 다음 수정은 하나만 정합니다.", style, PALE_BLUE),
                    box("5 - 같은 함수만 비교", "막힌 부분만 solution.py의 같은 함수와 비교합니다. 정답 전체는 닫아 둡니다.", style, PALE_TEAL),
                    box("6 - 변수 하나 바꾸기", "seed, top-k, threshold, batch size 중 하나를 바꾸고 결과를 설명합니다.", style, PALE_AMBER),
                ]
            ),
            Spacer(1, 8 * mm),
            para("실습 기록 템플릿", style["h2"]),
        ]
    )
    note = Table(
        [[para("가설:<br/>내가 직접 쓴 코드:<br/>실행 결과 또는 오류:<br/>다음에 바꿀 한 가지:", style["body"])]],
        colWidths=[174 * mm],
    )
    note.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend([note, PageBreak()])

    story.extend(
        [
            para("3. NLP 프로젝트 - 직접 만드는 여섯 단계", style["h1"]),
            para(
                "허브의 도구와 자료 탭에서 타이핑 워크북을 열고 원하는 프로젝트를 선택하세요. "
                "각 프로젝트는 starter.py와 solution.py의 구조가 같으므로 starter.py를 직접 완성한 뒤 "
                "같은 함수만 비교하면 됩니다.",
                style["body"],
            ),
            grid(
                [
                    box("01 - 텍스트 전처리", "정규화, 토큰 경계, 데이터 계약을 구현합니다. 입력을 신뢰할 수 있는 형태로 바꿉니다.", style, PALE_BLUE),
                    box("02 - 의도 분류", "TF-IDF 기준선과 평가 지표를 만듭니다. accuracy와 macro F1을 비교합니다.", style, PALE_TEAL),
                    box("03 - 의미 검색", "벡터화, 유사도, top-k 검색을 조립합니다. 점수와 실제 관련성을 비교합니다.", style, PALE_AMBER),
                    box("04 - 직접 RAG", "문서 로드, 검색, grounded prompt를 잇습니다. 생성보다 근거를 먼저 봅니다.", style, PALE_BLUE),
                    box("05 - LangChain RAG", "문서, 청크, retriever의 경계를 의식하며 프레임워크 흐름을 조립합니다.", style, PALE_TEAL),
                    box("06 - RAG 평가", "Hit@k, Recall@k, MRR, coverage를 분리해 계산하고 실패 사례를 찾습니다.", style, PALE_AMBER),
                ]
            ),
            Spacer(1, 8 * mm),
            para("프로젝트 실행 명령", style["h2"]),
        ]
    )
    command = Table(
        [[para(r".\.venv\Scripts\python.exe projects\nlp\01_text_preprocessing\starter.py", style["code"])]],
        colWidths=[174 * mm],
    )
    command.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([command, PageBreak()])

    story.extend(
        [
            para("4. 막혔을 때의 순서", style["h1"]),
            grid(
                [
                    box("정답은 체크포인트", "정답은 시작점이 아니라 확인 지점입니다. 내 버전을 실행한 뒤에만 같은 함수와 비교합니다.", style, PALE_TEAL),
                    box("오류는 힌트", "traceback의 첫 번째 내 코드 줄, 입력 shape, 반환형을 확인합니다.", style, PALE_BLUE),
                    box("비교는 최소 단위", "solution.py 전체를 열지 말고 TODO 하나의 함수 이름만 찾아 봅니다.", style, PALE_AMBER),
                ]
            ),
            Spacer(1, 8 * mm),
            para("문제가 생겼을 때", style["h2"]),
        ]
    )
    help_data = [
        [para("증상", style["table_head"]), para("먼저 확인할 것", style["table_head"]), para("다음 행동", style["table_head"])],
        [para("실습 버튼이 실행되지 않음", style["table_cell"]), para("환경 관리 탭에서 준비 완료 상태인지", style["table_cell"]), para("설치 또는 복구 후 전체 환경 점검 실행", style["table_cell"])],
        [para("starter.py가 TODO에서 멈춤", style["table_cell"]), para("가장 위의 TODO 번호와 함수 시그니처", style["table_cell"]), para("해당 함수만 작은 입력으로 구현 후 재실행", style["table_cell"])],
        [para("출력이나 metric이 이상함", style["table_cell"]), para("입력 예시 하나, shape, label 분포, seed", style["table_cell"]), para("조건 하나만 바꿔 다시 실행하고 결과 기록", style["table_cell"])],
        [para("VS Code에서 Markdown이 열림", style["table_cell"]), para("허브의 읽기 쉬운 가이드 또는 PDF 버튼", style["table_cell"]), para("문서는 브라우저/PDF로, 코드는 VS Code에서 직접 입력", style["table_cell"])],
    ]
    help_table = Table(help_data, colWidths=[42 * mm, 63 * mm, 69 * mm])
    help_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            help_table,
            Spacer(1, 10 * mm),
            para(
                "오늘의 목표는 많이 보는 것이 아니라, 한 가지를 직접 쓰고 실행해 "
                "왜 동작하는지 설명할 수 있게 되는 것입니다.",
                style["quote"],
            ),
        ]
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"CREATED {OUTPUT}")


if __name__ == "__main__":
    build()
