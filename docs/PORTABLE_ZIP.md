# Windows 휴대용 ZIP 사용 안내

[← 저장소 학습 안내로 돌아가기](../README.md)

휴대용 ZIP은 GitHub 저장소에 포함되는 프로젝트 파일과 아직 커밋하지 않은 최신 로컬 변경을
한 번에 다른 Windows PC로 옮기기 위한 패키지입니다. 노트북, Python 코드, 학습 데이터,
한국어 논문 요약, Windows 실행기, requirements와 MIT 라이선스를 포함합니다.

## ZIP에 포함하지 않는 항목

다음 항목은 PC마다 다시 만들거나 보안상 공유하지 않아야 하므로 제외합니다.

- `.git/`: Git 내부 이력과 로컬 설정
- `.venv/`, `venv/`: 현재 PC와 Python 설치에 종속된 가상환경
- `.env`, `.streamlit/secrets.toml`: API key와 비밀 설정
- `__pycache__/`, pytest·Ruff·mypy cache: 다시 생성되는 cache
- model checkpoint, log, 임시 artifact, Jupyter checkpoint
- `requirements-local-lock.txt`: 현재 PC에서만 쓰는 선택적 로컬 lock

따라서 ZIP 하나만 복사하면 프로젝트 자료는 모두 옮겨지지만, Python과 GPU driver까지
복사되는 것은 아닙니다. 새 PC에는 **64비트 Python 3.12**가 필요하고, 첫 의존성 설치 때는
인터넷 연결이 필요합니다. 설치 스크립트를 한 번 실행하면 PC 전용 `.venv`와 Jupyter
kernel이 새로 만들어집니다.

공유 과정에서 개인 경로나 장치 정보가 섞이지 않도록, 기본 패키지는 노트북의 **코드와
Markdown은 그대로 보존하고 저장된 출력과 실행 번호만 압축 사본에서 제거**합니다. 원래
작업 폴더의 노트북은 수정하지 않습니다. 출력까지 꼭 전달해야 할 때만 패키지 생성 시
`-IncludeNotebookOutputs` 옵션을 명시적으로 사용하세요. 이 옵션으로 만든 ZIP은 저장 출력과
실행 번호 때문에 `Verify_Setup.cmd`의 소스 청결도 검사가 실패할 수 있으며, 실행 결과를
의도적으로 보관하는 경우에만 사용해야 합니다.

## 다른 Windows PC에서 시작하는 순서

1. 새 PC에 **64비트 Python 3.12**를 설치하고 인터넷 연결을 확인합니다.
2. ZIP의 SHA-256을 아래 방법으로 확인한 뒤 차단을 해제합니다.
3. ZIP을 `C:\ai-paper-lab` 또는 `%USERPROFILE%\ai-paper-lab`처럼 짧은 최종 폴더에 풉니다.
   OneDrive 동기화 폴더나 지나치게 긴 경로는 처음 설치할 때 피하는 편이 안전합니다.
4. ZIP 내부에서 직접 실행하지 말고 **압축을 완전히 푼 폴더**를 엽니다.
5. `Install_or_Repair.cmd`를 실행합니다.
6. NVIDIA GPU가 있다면 설치 완료 후 `Install_GPU_PyTorch.cmd`를 실행합니다.
7. `Verify_Setup.cmd`를 실행해 Python, 데이터, 노트북과 런처를 검사합니다.
8. 바탕화면 바로가기를 만들려면 저장소 루트의 PowerShell에서 다음을 실행합니다.

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\create_learning_shortcuts.ps1
   ```

9. 첫 학습은 `Start_Paired_Learning.cmd 00`으로 시작합니다.
10. 논문 실습은 `Start_Paper_Reproductions.cmd 00`, 분야별 실습은
   `Start_Field_Paper_Labs.cmd`를 사용합니다.

설치 후 프로젝트 폴더를 옮기면 가상환경·Jupyter kernel·바로가기 안의 경로가 달라집니다.
최종 위치로 옮긴 뒤 `Install_or_Repair.cmd`와 바로가기 생성 명령을 다시 실행하세요.

Windows가 인터넷에서 받은 파일을 차단하면 ZIP을 풀기 전에 파일을 우클릭하고
`속성 → 차단 해제 → 적용`을 선택합니다. 조직 PC에서 실행 정책이 제한되어 있으면 `.cmd`
파일을 우선 사용하고, PowerShell 스크립트 실행에는 관리자 정책을 따릅니다.

## 무결성 확인

압축 안의 `SHA256SUMS.txt`에는 프로젝트 payload와 `PORTABLE_PACKAGE_INFO.md`의 SHA-256이
기록됩니다. `PORTABLE_PACKAGE_INFO.md`에는 패키지를 만든 branch·commit·로컬 변경 수와
첫 실행 순서가 들어 있습니다.

ZIP을 만들 때 스크립트는 다음을 자동 검사합니다.

- ZIP entry에 절대경로나 `..` 경로가 없는지
- 대소문자를 무시했을 때 중복되는 Windows 경로가 없는지
- 모든 압축 stream의 SHA-256이 내부 manifest와 일치하는지
- 포함 파일이 Git의 tracked·untracked·non-ignored 범위와 정확히 일치하는지
- 정규 학습 노트북이 239개이며 금지된 cache·checkpoint가 없는지
- `.env.*`, 개인키·인증서처럼 비밀정보일 가능성이 큰 파일명이 없는지
- output과 임시 staging 경로가 저장소의 `artifacts/` 안에 있는지

다운로드한 ZIP 자체의 SHA-256은 Windows PowerShell에서 다음처럼 확인할 수 있습니다.

```powershell
Get-FileHash .\ai-paper-reproduction-study-windows-portable-*.zip -Algorithm SHA256
```

표시된 값이 전달받은 SHA-256과 같은지 확인하세요.

새 패키지를 다시 만들려면 저장소 루트의 PowerShell에서 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\create_portable_zip.ps1
```

생성 결과는 기본적으로 `artifacts\releases\`에 저장됩니다. 같은 이름의 ZIP을 덮어쓰지
않으며 파일명에 생성 시각을 넣습니다.

## GitHub와의 차이

ZIP에는 현재 작업 폴더의 최신 내용이 들어가므로 아직 GitHub에 push하지 않은 변경도
포함될 수 있습니다. 반면 `.git/`은 제외되므로 ZIP을 푼 폴더 자체는 Git clone이 아닙니다.
다른 PC에서도 Git 이력과 원격 업데이트가 필요하다면 GitHub 저장소를 별도로 clone하고,
이 ZIP은 **소스·학습 데이터의 오프라인 백업** 또는 현재 작업본 전달용으로 사용하세요.
단, 새 PC에서 Python 의존성을 처음 설치하는 과정은 인터넷 연결이 필요합니다.
