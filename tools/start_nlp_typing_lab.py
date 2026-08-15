"""Prepare a non-destructive personal NLP starter file and optionally use it."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "01": "01_text_preprocessing",
    "02": "02_intent_classification",
    "03": "03_semantic_search",
    "04": "04_rag_from_scratch",
    "05": "05_langchain_rag",
    "06": "06_rag_evaluation",
}


def resolve_project(project: str) -> tuple[Path, Path]:
    normalized = project.strip().zfill(2)
    try:
        folder_name = PROJECTS[normalized]
    except KeyError as error:
        choices = ", ".join(PROJECTS)
        raise ValueError(f"프로젝트 번호는 {choices} 중 하나여야 합니다.") from error

    original = ROOT / "projects" / "nlp" / folder_name / "starter.py"
    personal = ROOT / "learner_work" / "nlp" / folder_name / "starter.py"
    if not original.is_file():
        raise FileNotFoundError(f"원본 starter.py를 찾지 못했습니다: {original}")
    return original, personal


def prepare_personal_starter(project: str, *, dry_run: bool = False) -> Path:
    original, personal = resolve_project(project)
    if personal.exists():
        print(f"KEEP {personal}")
        return personal
    if dry_run:
        print(f"WOULD_COPY {original} -> {personal}")
        return personal

    personal.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, personal)
    note_path = personal.with_name("README_FIRST.txt")
    note_path.write_text(
        "\n".join(
            [
                "이 폴더는 원본을 건드리지 않는 개인 타이핑 작업본입니다.",
                "",
                "1. starter.py의 TODO 하나를 직접 타이핑합니다.",
                "2. 저장한 뒤 프로젝트 루트에서 starter.py를 실행합니다.",
                "3. 막힌 함수 하나만 solution.py와 비교합니다.",
                "",
                "이 파일은 자동으로 덮어쓰지 않습니다.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"CREATED {personal}")
    return personal


def find_vscode() -> str | None:
    for command in ("code", "code.cmd", "code.exe"):
        located = shutil.which(command)
        if located:
            return located
    candidates = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "Microsoft VS Code"
        / "Code.exe",
        Path(os.environ.get("ProgramFiles", ""))
        / "Microsoft VS Code"
        / "Code.exe",
        Path(os.environ.get("ProgramFiles(x86)", ""))
        / "Microsoft VS Code"
        / "Code.exe",
    )
    return next((str(path) for path in candidates if path.is_file()), None)
    return None


def open_in_editor(path: Path) -> None:
    code = find_vscode()
    if code:
        subprocess.Popen([code, str(path)], cwd=str(ROOT))
        print(f"OPENED_VSCODE {path}")
        return
    if os.name == "nt":
        subprocess.Popen(["notepad.exe", str(path)], cwd=str(ROOT))
        print(f"OPENED_NOTEPAD {path}")
        return
    raise RuntimeError("VS Code 명령을 찾지 못했습니다. 파일 경로를 직접 열어 주세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", default="01", help="프로젝트 번호 01~06")
    parser.add_argument("--open-editor", action="store_true", help="개인 starter.py를 코드 편집기로 엽니다.")
    parser.add_argument("--run", action="store_true", help="개인 starter.py를 현재 Python으로 실행합니다.")
    parser.add_argument("--dry-run", action="store_true", help="복사 대상만 표시하고 파일을 만들지 않습니다.")
    args = parser.parse_args()

    personal = prepare_personal_starter(args.project, dry_run=args.dry_run)
    if args.dry_run:
        return
    if args.open_editor:
        open_in_editor(personal)
    if args.run:
        raise SystemExit(subprocess.call([sys.executable, str(personal)], cwd=ROOT))
    if not args.open_editor and not args.run:
        print(f"READY {personal}")
        print("다음: --open-editor로 직접 타이핑하거나 --run으로 실행하세요.")


if __name__ == "__main__":
    main()
