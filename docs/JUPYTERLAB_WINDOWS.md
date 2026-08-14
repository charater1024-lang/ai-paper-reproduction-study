# Windows JupyterLab 사용 안내

## 실습과 정답을 좌우 2창으로 열기

바탕화면의 **AI 코딩연습 - 기본 실습과 정답**을 두 번 클릭하고 실습 번호를 입력합니다.
왼쪽에는 `notebooks/exercises/`, 오른쪽에는 `notebooks/solutions/`의 같은 파일이 열립니다.
Edge 또는 Chrome이 설치되어 있으면 현재 Windows 작업 영역을 감지해 절반 크기의 별도
창 두 개를 요청합니다. 브라우저가 위치를 기억해 겹치면 두 창을 `Win+←`, `Win+→`로 정렬하세요.

서버는 하나만 실행되므로 두 창이 같은 `Python (AI Engineering Lab)` 커널 환경을
사용하지만 각 노트북의 kernel 상태는 독립적입니다. 종료할 때 실행 콘솔에서 `Ctrl+C`를
누르면 서버가 함께 정리됩니다.

```powershell
.\Start_Paired_Learning.cmd       # 번호 선택 메뉴
.\Start_Paired_Learning.cmd 17    # Embedding/Vector Index 바로 열기
```

## 가장 쉬운 실행

학습 폴더의 `Start_JupyterLab.cmd`를 실행합니다. 브라우저가 열리면 왼쪽 `notebooks` 폴더에서
`00_`부터 순서대로 진행합니다.

노트북 오른쪽 위 커널 이름이 `Python (AI Engineering Lab)`인지 확인하세요. 다른 커널이
보이면 커널 이름을 클릭해 교체합니다.

PyTorch 학습은 CUDA → Apple MPS → CPU를 자동 선택합니다. setup 셀의 `device=` 출력이
`cuda`인지 확인하는 방법, CPU 강제 비교, AMP와 GPU 메모리 문제는
[GPU 자동 가속 사용 안내](GPU_ACCELERATION.md)를 참고하세요.

## 화면을 단순하게 쓰고 싶을 때

JupyterLab의 `View → Simple Interface Mode`를 켜면 한 노트북에만 집중할 수 있습니다.
코드 파일, 데이터, 터미널을 동시에 보고 싶을 때 다시 끕니다.

## 권장 패널 배치

- 왼쪽: 노트북
- 오른쪽 위: `src/`의 실제 구현 코드
- 오른쪽 아래: PowerShell Terminal 또는 CSV/JSON 결과
- 왼쪽 사이드바 `Running`: 살아 있는 kernel과 terminal 확인

## 노트북 실행 규칙

1. 처음 열었으면 `Kernel → Restart Kernel and Run All Cells`로 전체 실행합니다.
2. 셀을 수정할 때는 한 번에 한 조건만 변경합니다.
3. 실행 번호가 뒤섞였으면 커널을 재시작합니다.
4. 노트북 코드가 길어지면 `src/llm_engineering_lab/`로 옮기고 테스트를 추가합니다.
5. 작업 후 `Kernel → Shut Down Kernel`로 GPU/메모리를 해제합니다.

## 복구

패키지가 없다는 오류가 나면 `Install_or_Repair.cmd`를 실행합니다. 전역 Python이나 Jupyter
설정을 변경하지 않고 이 폴더의 `.venv`만 복구합니다.

NVIDIA GPU용 PyTorch를 다시 설치하려면 `Install_GPU_PyTorch.cmd`를 실행합니다. 기본
노트북은 CUDA가 없어도 CPU에서 동작하며, GPU 설치 결과는 `Verify_Setup.cmd`로 확인합니다.

학습 폴더에서 PowerShell을 연 뒤 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\python.exe -m pytest -q
```

커널 목록 확인:

```powershell
.\.venv\Scripts\python.exe -m jupyter kernelspec list
```

## 종료

브라우저 탭만 닫으면 kernel과 서버가 계속 실행될 수 있습니다. JupyterLab 터미널 창에서
`Ctrl+C`를 두 번 누르고 종료 여부에 답하거나, `File → Shut Down`을 사용합니다.
