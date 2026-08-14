# Windows 설치 및 다른 PC에서 실행하기

이 문서는 GitHub에서 프로젝트를 새 Windows PC로 받은 뒤 같은 학습 환경을 만드는 절차입니다. 기준 환경은 **64비트 Windows 10/11과 CPython 3.12**입니다. NVIDIA GPU는 선택 사항이며, GPU가 없어도 모든 교육용 미니 실험을 CPU로 실행할 수 있습니다.

## 가장 빠른 설치

1. [Python 3.12](https://www.python.org/downloads/windows/) 64비트 버전을 설치합니다. 설치 화면에서 `Add python.exe to PATH`를 선택합니다.
2. GitHub 저장소를 Clone하거나 ZIP으로 내려받아 압축을 풉니다.
3. 프로젝트 루트의 `Install_or_Repair.cmd`를 더블클릭합니다.
4. 설치가 끝나면 `Verify_Setup.cmd`를 실행합니다.
5. `Start_JupyterLab.cmd` 또는 원하는 분야별 실행기를 실행합니다.

`Install_or_Repair.cmd`는 `.venv`를 만들고 공식 **CPU 전용 PyTorch**와 학습 패키지를 설치합니다. 이미 정상적인 CUDA PyTorch가 설치되어 있다면 해당 빌드를 유지합니다.

CUDA 빌드를 CPU 전용 빌드로 명시적으로 되돌릴 때는 다음 옵션을 사용합니다.

```powershell
.\Install_or_Repair.cmd --cpu
```

> 저장소를 `C:\ai-paper-lab`처럼 짧은 경로에 두면 OneDrive 동기화나 매우 긴 파일 경로로 인한 문제를 줄일 수 있습니다. 현재처럼 한글·공백이 포함된 경로도 배치 파일에서 지원합니다.

## PowerShell에서 수동 설치

활성화 스크립트를 사용하지 않고 `.venv`의 Python을 직접 호출하므로 PowerShell 실행 정책을 바꿀 필요가 없습니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --editable . --no-deps
.\.venv\Scripts\python.exe -m ipykernel install --user `
  --name ai-engineering-lab `
  --display-name "Python (AI Engineering Lab)"
```

Python Launcher(`py`)가 없다면 첫 줄만 설치된 Python 3.12의 전체 경로로 바꿉니다.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
```

## NVIDIA GPU 사용

먼저 CPU 기본 설치를 완료한 다음 NVIDIA 드라이버를 최신 상태로 갱신하고 다음 파일을 실행합니다.

```powershell
.\Install_GPU_PyTorch.cmd
```

기본 채널은 PyTorch 2.13.0의 CUDA 13.0(`cu130`)입니다. 드라이버 호환성 때문에 더 낮은 CUDA 런타임이 필요하면 다음처럼 공식 CUDA 12.6 빌드를 선택할 수 있습니다.

```powershell
.\Install_GPU_PyTorch.cmd cu126
```

CUDA 13.2 채널이 필요한 최신 환경에서는 `cu132`도 선택할 수 있습니다. PyTorch wheel에 CUDA 런타임이 포함되므로 일반적인 노트북 실습에는 별도 CUDA Toolkit 설치가 필요하지 않습니다. NVIDIA 디스플레이 드라이버는 반드시 필요합니다.

설치 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

노트북은 기본적으로 `CUDA → CPU` 순서로 장치를 자동 선택합니다. 일시적으로 CPU 실행을 강제하려면 같은 PowerShell 창에서 다음과 같이 실행합니다.

```powershell
$env:AI_LAB_DEVICE = "cpu"
.\Start_JupyterLab.cmd
```

## 의존성 파일의 역할

| 파일 | 용도 |
|---|---|
| `requirements.txt` | PyTorch를 제외한 공통 학습·실행 패키지와 호환 버전 범위 |
| `requirements-windows.txt` | 공통 패키지 + Windows/Python 3.12용 공식 CPU PyTorch |
| `requirements-dev.txt` | 테스트와 코드 검사 도구 |
| `pyproject.toml` | 프로젝트 자체 패키지와 선택 기능의 메타데이터 |

NumPy 같은 일반 패키지는 호환되는 버전 범위로 관리하고, 하드웨어 빌드가 중요한 Windows PyTorch만 명시적으로 고정했습니다. 따라서 특정 PC에서 생성한 거대한 `pip freeze` 결과를 공유하는 것보다 다른 Windows PC에서 재현하기 쉽습니다.

개인 실험을 완전히 동일한 버전으로 보관해야 할 때만 별도 파일을 만들 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-local-lock.txt
```

`requirements-local-lock.txt`는 GPU 종류와 설치 시점에 종속될 수 있으므로 일반 배포용 파일로 사용하지 않는 것을 권장합니다.

## 데이터

교육용 데이터는 저장소의 `data` 폴더에 포함되어 있어 기본 실습에 별도 다운로드나 API 키가 필요하지 않습니다. 데이터가 손상되었거나 직접 다시 만들고 싶다면 설치 후 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts\07_generate_practice_datasets.py
.\.venv\Scripts\python.exe tools\generate_field_datasets.py
```

생성 전 기존 파일을 보존해야 한다면 `data` 폴더를 먼저 복사해 두세요.

## 문제 해결

### Python 3.12를 찾지 못함

`python --version` 또는 `py -3.12 --version`을 확인합니다. Microsoft Store 별칭이 다른 Python으로 연결되면 python.org의 64비트 Python 3.12를 설치하고 터미널을 새로 엽니다.

### 기존 `.venv` 버전이 다름

열려 있는 Jupyter와 Python 프로세스를 종료한 뒤 `.venv`를 `.venv-backup`으로 이름 변경하고 `Install_or_Repair.cmd`를 다시 실행합니다. 새 환경이 정상 작동하는지 확인한 후 백업을 정리합니다.

### GPU 설치 후 `torch.cuda.is_available()`이 `False`

`nvidia-smi`가 정상 실행되는지 먼저 확인합니다. 드라이버를 갱신한 후 재부팅하고 GPU 설치기를 다시 실행합니다. 오래된 드라이버를 유지해야 한다면 `Install_GPU_PyTorch.cmd cu126`을 시도합니다.

GPU 사용을 중단하고 검증된 CPU 빌드로 복원하려면 `Install_or_Repair.cmd --cpu`를 실행합니다.

### Jupyter에서 이전 커널이 선택됨

노트북의 `Kernel` 메뉴에서 `Python (AI Engineering Lab)`을 선택하고 커널을 재시작합니다. 그래도 해결되지 않으면 `Install_or_Repair.cmd`가 커널을 다시 등록합니다.

### 설치 상태 전체 확인

```powershell
.\Verify_Setup.cmd
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

PyTorch Windows 지원 범위와 최신 설치 채널은 [PyTorch 공식 Windows 설치 안내](https://docs.pytorch.org/get-started/locally/)에서 확인할 수 있습니다.
