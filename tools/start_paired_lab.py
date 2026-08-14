"""Start one JupyterLab server and open exercise/solution in two browser windows."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import quote

if os.name == "nt":
    from ctypes import wintypes

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "notebooks" / "exercises"
SOLUTIONS = ROOT / "notebooks" / "solutions"
PAPER_EXERCISES = ROOT / "notebooks" / "paper_reproductions" / "exercises"
PAPER_SOLUTIONS = ROOT / "notebooks" / "paper_reproductions" / "solutions"
FIELD_TRACK = ROOT / "notebooks" / "field_reproductions"
FIELD_IDS = (
    "vision",
    "nlp_llm",
    "generative",
    "reinforcement_learning",
    "graph_recommendation",
    "self_supervised_multimodal",
    "distillation_compression",
)
LOG_DIRECTORY = ROOT / "artifacts"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def configure_runtime_kernelspec() -> Path:
    """Expose a project-local kernel bound to the current interpreter.

    This avoids stale user kernelspecs after the repository is moved and does not mutate the
    user's global Jupyter configuration. Each launcher gets its own temporary runtime tree so
    simultaneous starts cannot contend for Jupyter's cookie-secret and runtime files.
    """

    runtime_root = Path(tempfile.mkdtemp(prefix="ai-paper-lab-"))
    kernel_directory = runtime_root / "kernels" / "ai-engineering-lab"
    kernel_directory.mkdir(parents=True, exist_ok=True)
    runtime_directory = runtime_root / "runtime"
    config_directory = runtime_root / "config"
    ipython_directory = runtime_root / "ipython"
    for directory in (runtime_directory, config_directory, ipython_directory):
        directory.mkdir(parents=True, exist_ok=True)
    kernel = {
        "argv": [
            str(Path(sys.executable).resolve()),
            "-Xfrozen_modules=off",
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Python (AI Engineering Lab)",
        "language": "python",
        "metadata": {"debugger": True},
        "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    }
    (kernel_directory / "kernel.json").write_text(
        json.dumps(kernel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    existing = os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), existing) if value
    )
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_directory)
    os.environ["JUPYTER_CONFIG_DIR"] = str(config_directory)
    os.environ["IPYTHONDIR"] = str(ipython_directory)
    return runtime_root


def assign_kill_on_close_job(process: subprocess.Popen) -> int | None:
    """Put Jupyter in a Windows job so closing this launcher also closes its children."""

    if os.name != "nt":
        return None

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operations", ctypes.c_ulonglong),
            ("write_operations", ctypes.c_ulonglong),
            ("other_operations", ctypes.c_ulonglong),
            ("read_bytes", ctypes.c_ulonglong),
            ("write_bytes", ctypes.c_ulonglong),
            ("other_bytes", ctypes.c_ulonglong),
        ]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_longlong),
            ("per_job_user_time_limit", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", BasicLimitInformation),
            ("io_info", IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    job_handle = kernel32.CreateJobObjectW(None, None)
    if not job_handle:
        return None
    information = ExtendedLimitInformation()
    information.basic_limit_information.limit_flags = 0x00002000  # KILL_ON_JOB_CLOSE
    configured = kernel32.SetInformationJobObject(
        job_handle,
        9,  # JobObjectExtendedLimitInformation
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        job_handle, wintypes.HANDLE(process._handle)
    )
    if not assigned:
        kernel32.CloseHandle(job_handle)
        return None
    return int(job_handle)


def close_job(job_handle: int | None) -> None:
    if job_handle is not None and os.name == "nt":
        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(job_handle))


def available_pairs(
    exercises: Path = EXERCISES, solutions: Path = SOLUTIONS
) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    for exercise_path in sorted(exercises.glob("[0-9][0-9]_*.ipynb")):
        solution_path = solutions / exercise_path.name
        if solution_path.exists():
            pairs[exercise_path.name[:2]] = (exercise_path, solution_path)
    return pairs


def select_pair(
    number: str | None,
    pairs: dict[str, tuple[Path, Path]],
    default_number: str = "13",
) -> str:
    if number is not None:
        normalized = number.strip().zfill(2)
        if normalized not in pairs:
            raise SystemExit(f"[ERROR] {normalized}번 실습/정답 파일 쌍이 없습니다.")
        return normalized

    print("\n사용 가능한 2창 실습")
    for key, (exercise_path, _) in pairs.items():
        title = exercise_path.stem[3:].replace("_", " ")
        print(f"  {key}: {title}")
    answer = (
        input(
            f"\n실습 번호를 입력하세요 (한 자리도 가능, 예: 9 또는 09) "
            f"[기본 {default_number}]: "
        ).strip()
        or default_number
    )
    normalized = answer.zfill(2)
    if normalized not in pairs:
        raise SystemExit(f"[ERROR] {normalized}번 실습/정답 파일 쌍이 없습니다.")
    return normalized


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def browser_executable() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    return next((path for path in candidates if path.exists()), None)


def screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except (AttributeError, OSError):
        return 1920, 1080


def notebook_url(port: int, token: str, path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    return f"http://127.0.0.1:{port}/lab/tree/{quote(relative, safe='/')}?token={token}"


def open_two_windows(exercise_url: str, solution_url: str) -> None:
    executable = browser_executable()
    if executable is None:
        webbrowser.open_new(exercise_url)
        time.sleep(1)
        webbrowser.open_new(solution_url)
        print(
            "기본 브라우저를 사용했습니다. 창이 탭으로 열리면 탭 하나를 밖으로 끌어내세요."
        )
        return

    width, height = screen_size()
    half = max(width // 2, 800)
    common = [
        str(executable),
        "--new-window",
        "--no-first-run",
        f"--window-size={half},{height}",
    ]
    subprocess.Popen(
        [*common, "--window-position=0,0", exercise_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    subprocess.Popen(
        [*common, f"--window-position={half},0", solution_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until_ready(
    port: int,
    token: str,
    process: subprocess.Popen,
    log_path: Path,
    timeout: int = 45,
) -> None:
    health_url = f"http://127.0.0.1:{port}/api?token={token}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"JupyterLab이 예기치 않게 종료되었습니다. 로그: {log_path}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError(f"JupyterLab 시작 시간이 초과되었습니다. 로그: {log_path}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def request_server_shutdown(port: int, token: str, process: subprocess.Popen) -> None:
    """Ask Jupyter to stop kernels first, then fall back to terminating the process."""

    if process.poll() is not None:
        return
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/shutdown?token={token}",
        data=b"",
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
        process.wait(timeout=10)
    except (urllib.error.URLError, TimeoutError, subprocess.TimeoutExpired):
        stop_process(process)


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", nargs="?", help="실습 번호, 생략하면 메뉴 표시")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--paper",
        action="store_true",
        help="기본 커리큘럼 대신 20편 논문 미니 재현 트랙(00~19)을 엽니다",
    )
    source_group.add_argument(
        "--field",
        choices=FIELD_IDS,
        help="분야별 10편 논문 트랙 중 하나를 엽니다",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="서버 응답만 확인하고 종료"
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if args.field:
        exercises = FIELD_TRACK / args.field / "exercises"
        solutions = FIELD_TRACK / args.field / "solutions"
    elif args.paper:
        exercises = PAPER_EXERCISES
        solutions = PAPER_SOLUTIONS
    else:
        exercises = EXERCISES
        solutions = SOLUTIONS
    pairs = available_pairs(exercises, solutions)
    if not pairs:
        raise SystemExit(
            f"[ERROR] {exercises}와 {solutions}에서 노트북 쌍을 찾지 못했습니다."
        )
    default_number = "00" if (args.paper or args.field) else "13"
    number = select_pair(
        args.number or (next(iter(pairs)) if args.smoke_test else None),
        pairs,
        default_number=default_number,
    )
    runtime_root = configure_runtime_kernelspec()
    exercise_path, solution_path = pairs[number]
    port = find_free_port()
    token = secrets.token_urlsafe(24)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIRECTORY / f"paired_jupyter_{port}_{os.getpid()}.log"

    command = [
        sys.executable,
        "-m",
        "jupyterlab",
        "--no-browser",
        "--ServerApp.ip=127.0.0.1",
        f"--ServerApp.port={port}",
        "--ServerApp.port_retries=0",
        f"--ServerApp.token={token}",
        f"--ServerApp.root_dir={ROOT}",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        job_handle = assign_kill_on_close_job(process)
        try:
            wait_until_ready(port, token, process, log_path)
            if args.smoke_test:
                print(
                    f"PASS: paired JupyterLab server HTTP 200 (pair={number}, port={port})"
                )
                return

            exercise_url = notebook_url(port, token, exercise_path)
            solution_url = notebook_url(port, token, solution_path)
            open_two_windows(exercise_url, solution_url)
            print("\n============================================================")
            if args.field:
                track_label = f"분야별 논문 · {args.field}"
            else:
                track_label = "논문 재현" if args.paper else "기본 커리큘럼"
            print(
                f"  2창 {track_label} {number}: {exercise_path.stem[3:].replace('_', ' ')}"
            )
            print("  왼쪽: 실습용 빈 코드    오른쪽: 정답 코드")
            print(f"  로그: {log_path.name}")
            print("  종료: 이 창에서 Ctrl+C")
            print("============================================================\n")
            while process.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nJupyterLab을 종료합니다...")
        finally:
            request_server_shutdown(port, token, process)
            close_job(job_handle)
            shutil.rmtree(runtime_root, ignore_errors=True)


if __name__ == "__main__":
    main()
