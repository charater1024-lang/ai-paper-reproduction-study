# Windows에서 GitHub에 처음 게시하기

이 문서는 이 저장소를 다른 Windows 기기에서도 그대로 설치할 수 있는 형태로 GitHub에
게시하는 절차입니다. 기본 브랜치는 `main`을 사용합니다.

## 게시 전 선택

GitHub 저장소를 만들기 전에 다음 세 가지를 정합니다.

1. 저장소 이름: `ai-paper-reproduction-study`
2. 공개 범위: `public`
3. 코드 라이선스: MIT

루트 [LICENSE](../LICENSE)는 코드와 직접 작성한 문서에 MIT License를 적용합니다. 분야별
합성 데이터인 `data/field_curriculum/`의 CC0-1.0 안내처럼 별도 조건이 명시된 자료에는 해당
조건이 우선합니다. 링크된 논문 원문 저작권은 각 저자와 출판사에 있습니다.

## 1. GitHub CLI 설치와 로그인

Windows 10/11 PowerShell에서 공식 WinGet 패키지를 설치합니다.

```powershell
winget install --id GitHub.cli --source winget
```

설치 후에는 기존 터미널의 새 탭이 아니라 **새 터미널 창**을 열고 로그인합니다.

```powershell
gh --version
gh auth login
gh auth status
```

## 2. 로컬 검증

저장소 루트에서 다음을 실행합니다.

```powershell
.\Verify_Setup.cmd

# 업로드 대상과 제외 대상을 확인
git status --short
git status --ignored --short

# 제외 규칙을 직접 확인
Get-Content .gitignore
```

`.venv/`, `.env`, `artifacts/`, `.ipynb_checkpoints/`, `__pycache__/`는 GitHub에 올리지 않습니다.
API 키, 개인 경로, 실제 개인정보가 보이면 commit 전에 제거합니다.

## 3. main 브랜치와 첫 commit

아직 commit이 없는 기존 로컬 저장소라면 다음처럼 진행합니다.

```powershell
git branch -m main
git add .
git status
git diff --cached --stat
git commit -m "Initial release: Windows AI paper learning lab"
```

`git status`에서 `.venv`, model checkpoint, 로그나 개인 secret 파일이 stage되지 않았는지 반드시
확인한 뒤 commit합니다.

## 4. 빈 GitHub 저장소 생성과 push

로컬에 이미 README와 `.gitignore`가 있으므로 GitHub 웹에서 README·`.gitignore`·License를
자동 생성한 저장소와 합치지 않습니다. CLI로 빈 저장소를 만들면 이 문제를 피할 수 있습니다.

```powershell
# 공개 저장소 예시
gh repo create ai-paper-reproduction-lab `
  --public `
  --source . `
  --remote origin `
  --push

# 비공개로 시작하려면 --public 대신 --private
```

명령이 끝난 뒤 확인합니다.

```powershell
git branch --show-current
git remote -v
git status -sb
gh repo view --web
```

## 5. 이후 변경은 branch와 Pull Request로 관리

첫 `main`을 만든 뒤에는 기능별 branch를 권장합니다.

```powershell
git switch -c docs/improve-learning-guide

# 파일 수정과 검증
git add README.md docs
git commit -m "Improve Korean learning guide"
git push -u origin docs/improve-learning-guide

gh pr create --draft --base main --fill
```

GitHub의 branch protection 또는 ruleset에서 `main` 직접 push를 막고 Windows CI 통과를
요구하면 실수로 깨진 자료가 기본 브랜치에 들어가는 것을 줄일 수 있습니다.

## 다른 Windows 기기에서 사용

```powershell
git clone https://github.com/사용자명/ai-paper-reproduction-lab.git
cd ai-paper-reproduction-lab
.\Install_or_Repair.cmd
.\Verify_Setup.cmd
.\Start_Paired_Learning.cmd 00
```

NVIDIA GPU를 쓸 기기에서는 기본 설치 후 다음을 추가로 실행합니다.

```powershell
.\Install_GPU_PyTorch.cmd
.\Verify_Setup.cmd
```

CPU와 GPU 설치 방법, 지원 Python 버전과 requirements 구성은
[Windows 설치 안내](WINDOWS_SETUP.md)를 참고합니다.

## 공개 저장소 체크리스트

- [ ] 루트 README 첫 화면만 읽어도 설치와 학습 시작 방법을 알 수 있다.
- [ ] `LICENSE`와 데이터 라이선스의 적용 범위를 구분했다.
- [ ] AI 보조 한국어 요약임을 밝히고 원 논문 링크를 제공한다.
- [ ] Windows CI가 `pytest`와 notebook 구조 검사를 통과한다.
- [ ] `.env`, API 키, checkpoint와 개인 로그가 제외되었다.
- [ ] 생성 notebook을 직접 고치지 않고 source spec/builder를 수정하는 규칙을 문서화했다.
- [ ] `main`에는 검증된 commit만 들어간다.
