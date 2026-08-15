#requires -Version 5.1
<#
.SYNOPSIS
Single-window launcher for the portable AI coding-practice package.

.DESCRIPTION
This script deliberately does not install Python packages or alter notebooks.
It presents the existing entry points in one Windows Forms hub and runs each
original launcher in its own visible console window when needed.
#>

[CmdletBinding()]
param(
    [switch]$ValidateOnly,

    [switch]$SmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$script:WorkspaceRoot = $PSScriptRoot
$script:PythonPath = Join-Path $script:WorkspaceRoot '.venv\Scripts\python.exe'
$script:CommandProcessor = (Get-Command 'cmd.exe' -ErrorAction Stop).Source
$script:MainForm = $null
$script:EnvironmentBadge = $null

function Show-HubMessage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,

        [Parameter()]
        [string]$Title = 'AI 코딩 실습 센터',

        [Parameter()]
        [System.Windows.Forms.MessageBoxIcon]$Icon = [System.Windows.Forms.MessageBoxIcon]::Information
    )

    [void][System.Windows.Forms.MessageBox]::Show(
        $script:MainForm,
        $Message,
        $Title,
        [System.Windows.Forms.MessageBoxButtons]::OK,
        $Icon
    )
}

function Test-HubFiles {
    $requiredItems = @(
        'Install_or_Repair.cmd',
        'Install_GPU_PyTorch.cmd',
        'Verify_Setup.cmd',
        'Start_JupyterLab.cmd',
        'Start_Paired_Learning.cmd',
        'Start_Paper_Reproductions.cmd',
        'Start_Field_Paper_Labs.cmd',
        'Start_NLP_Typing_Lab.cmd',
        'Open_Learning_Folder.cmd',
        'app.py',
        'AI_Coding_Practice_Guide.html',
        'AI_Coding_Practice_Guide.pdf',
        'AI_Korean_Paper_Notes.html',
        'AI_Function_Glossary.html',
        'projects\nlp\Typing_Practice_Workbook.html',
        'tools\start_nlp_typing_lab.py',
        'tools\build_function_glossary.py',
        'notebooks',
        'projects\nlp',
        'scripts',
        'data'
    )

    $missingItems = @(
        foreach ($relativePath in $requiredItems) {
            $fullPath = Join-Path $script:WorkspaceRoot $relativePath
            if (-not (Test-Path -LiteralPath $fullPath)) {
                $relativePath
            }
        }
    )

    if ($missingItems.Count -gt 0) {
        throw (
            '학습 패키지 파일을 찾지 못했습니다: ' +
            ($missingItems -join ', ') +
            '. 이 파일을 원래 학습 폴더의 최상위 위치에 두세요.'
        )
    }
}

function Test-LearningEnvironment {
    return Test-Path -LiteralPath $script:PythonPath -PathType Leaf
}

function Set-EnvironmentBadge {
    param(
        [Parameter()]
        [System.Windows.Forms.Label]$Badge = $script:EnvironmentBadge
    )

    if ($null -eq $Badge) {
        return
    }

    if (Test-LearningEnvironment) {
        $Badge.Text = '● 학습 환경 준비 완료'
        $Badge.ForeColor = [System.Drawing.Color]::FromArgb(24, 112, 72)
        $Badge.BackColor = [System.Drawing.Color]::FromArgb(224, 247, 235)
    }
    else {
        $Badge.Text = '● 첫 설치가 필요합니다'
        $Badge.ForeColor = [System.Drawing.Color]::FromArgb(141, 82, 0)
        $Badge.BackColor = [System.Drawing.Color]::FromArgb(255, 242, 204)
    }
}

function Require-LearningEnvironment {
    if (Test-LearningEnvironment) {
        return $true
    }

    $answer = [System.Windows.Forms.MessageBox]::Show(
        $script:MainForm,
        '아직 Python 가상환경(.venv)이 준비되지 않았습니다.' +
        [Environment]::NewLine +
        [Environment]::NewLine +
        '먼저 설치/복구를 실행해야 JupyterLab, 노트북 실습, Streamlit 앱을 사용할 수 있습니다.' +
        [Environment]::NewLine +
        [Environment]::NewLine +
        '지금 설치/복구를 시작할까요?',
        'AI 코딩 실습 센터',
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Information
    )

    if ($answer -eq [System.Windows.Forms.DialogResult]::Yes) {
        Start-CmdLauncher -LauncherName 'Install_or_Repair.cmd'
    }

    return $false
}

function Start-CmdLauncher {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LauncherName,

        [Parameter()]
        [string[]]$Arguments = @()
    )

    $launcherPath = Join-Path $script:WorkspaceRoot $LauncherName
    if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
        throw "실행 파일을 찾지 못했습니다: $LauncherName"
    }

    foreach ($argument in $Arguments) {
        if ($argument -notmatch '^[A-Za-z0-9_-]+$') {
            throw "허용되지 않은 실행 인수입니다: $argument"
        }
    }

    # cmd.exe is used intentionally. It is more reliable than a direct .cmd
    # shortcut when the package is placed under a Korean or OneDrive path.
    $invocation = '"' + $launcherPath + '"'
    if ($Arguments.Count -gt 0) {
        $invocation += ' ' + ($Arguments -join ' ')
    }

    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $script:CommandProcessor
    $processInfo.Arguments = '/d /s /c "' + $invocation + '"'
    $processInfo.WorkingDirectory = $script:WorkspaceRoot
    $processInfo.UseShellExecute = $true

    [void][System.Diagnostics.Process]::Start($processInfo)
}

function Start-StreamlitLab {
    if (-not (Require-LearningEnvironment)) {
        return
    }

    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $script:PythonPath
    $processInfo.Arguments = '-m streamlit run app.py'
    $processInfo.WorkingDirectory = $script:WorkspaceRoot
    $processInfo.UseShellExecute = $true

    [void][System.Diagnostics.Process]::Start($processInfo)
}

function Open-WorkspaceItem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $targetPath = Join-Path $script:WorkspaceRoot $RelativePath
    if (-not (Test-Path -LiteralPath $targetPath)) {
        throw "열 파일 또는 폴더를 찾지 못했습니다: $RelativePath"
    }

    Start-Process -FilePath $targetPath
}

function Get-ChoiceCode {
    param(
        [Parameter(Mandatory = $true)]
        [System.Windows.Forms.ComboBox]$ComboBox
    )

    $selected = [string]$ComboBox.SelectedItem
    if ([string]::IsNullOrWhiteSpace($selected) -or $selected -notmatch '\|') {
        throw '실습 항목을 선택하세요.'
    }

    return $selected.Split('|')[0].Trim()
}

function New-HubButton {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,

        [Parameter(Mandatory = $true)]
        [int]$Left,

        [Parameter(Mandatory = $true)]
        [int]$Top,

        [Parameter(Mandatory = $true)]
        [int]$Width,

        [Parameter()]
        [int]$Height = 38,

        [Parameter()]
        [System.Drawing.Color]$BackColor = [System.Drawing.Color]::FromArgb(33, 99, 175)
    )

    $button = New-Object System.Windows.Forms.Button
    $button.Text = $Text
    $button.Location = New-Object System.Drawing.Point($Left, $Top)
    $button.Size = New-Object System.Drawing.Size($Width, $Height)
    $button.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $button.FlatAppearance.BorderSize = 0
    $button.BackColor = $BackColor
    $button.ForeColor = [System.Drawing.Color]::White
    $button.Font = New-Object System.Drawing.Font(
        'Malgun Gothic',
        9,
        [System.Drawing.FontStyle]::Bold
    )
    $button.Cursor = [System.Windows.Forms.Cursors]::Hand
    return $button
}

function New-HubLabel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,

        [Parameter(Mandatory = $true)]
        [int]$Left,

        [Parameter(Mandatory = $true)]
        [int]$Top,

        [Parameter(Mandatory = $true)]
        [int]$Width,

        [Parameter()]
        [int]$Height = 25,

        [Parameter()]
        [float]$FontSize = 9,

        [Parameter()]
        [System.Drawing.Color]$ForeColor = [System.Drawing.Color]::FromArgb(63, 70, 82)
    )

    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.Location = New-Object System.Drawing.Point($Left, $Top)
    $label.Size = New-Object System.Drawing.Size($Width, $Height)
    $label.Font = New-Object System.Drawing.Font('Malgun Gothic', $FontSize)
    $label.ForeColor = $ForeColor
    $label.AutoEllipsis = $true
    return $label
}

function New-HubGroup {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,

        [Parameter(Mandatory = $true)]
        [int]$Left,

        [Parameter(Mandatory = $true)]
        [int]$Top,

        [Parameter(Mandatory = $true)]
        [int]$Width,

        [Parameter(Mandatory = $true)]
        [int]$Height
    )

    $group = New-Object System.Windows.Forms.GroupBox
    $group.Text = $Title
    $group.Location = New-Object System.Drawing.Point($Left, $Top)
    $group.Size = New-Object System.Drawing.Size($Width, $Height)
    $group.Font = New-Object System.Drawing.Font(
        'Malgun Gothic',
        10,
        [System.Drawing.FontStyle]::Bold
    )
    $group.ForeColor = [System.Drawing.Color]::FromArgb(31, 50, 81)
    return $group
}

function New-HubComboBox {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Items,

        [Parameter(Mandatory = $true)]
        [int]$Left,

        [Parameter(Mandatory = $true)]
        [int]$Top,

        [Parameter(Mandatory = $true)]
        [int]$Width
    )

    $comboBox = New-Object System.Windows.Forms.ComboBox
    $comboBox.Location = New-Object System.Drawing.Point($Left, $Top)
    $comboBox.Size = New-Object System.Drawing.Size($Width, 30)
    $comboBox.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
    $comboBox.Font = New-Object System.Drawing.Font('Malgun Gothic', 9)

    foreach ($item in $Items) {
        [void]$comboBox.Items.Add($item)
    }
    $comboBox.SelectedIndex = 0
    return $comboBox
}

Test-HubFiles

if ($ValidateOnly) {
    Write-Output "VALID: AI 코딩 실습 센터의 필수 파일과 실행 경로를 확인했습니다."
    exit 0
}

$form = New-Object System.Windows.Forms.Form
$script:MainForm = $form
$form.Text = 'AI 코딩 실습 센터'
$form.ClientSize = New-Object System.Drawing.Size(1120, 760)
$form.MinimumSize = New-Object System.Drawing.Size(980, 700)
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.BackColor = [System.Drawing.Color]::FromArgb(246, 248, 252)
$form.Font = New-Object System.Drawing.Font('Malgun Gothic', 9)

$headerPanel = New-Object System.Windows.Forms.Panel
$headerPanel.Location = New-Object System.Drawing.Point(0, 0)
$headerPanel.Size = New-Object System.Drawing.Size(1120, 112)
$headerPanel.Anchor = (
    [System.Windows.Forms.AnchorStyles]::Top -bor
    [System.Windows.Forms.AnchorStyles]::Left -bor
    [System.Windows.Forms.AnchorStyles]::Right
)
$headerPanel.BackColor = [System.Drawing.Color]::FromArgb(31, 50, 81)
$form.Controls.Add($headerPanel)

$titleLabel = New-HubLabel -Text 'AI 코딩 실습 센터' -Left 26 -Top 18 -Width 430 -Height 38 -FontSize 20 -ForeColor ([System.Drawing.Color]::White)
$titleLabel.Font = New-Object System.Drawing.Font(
    'Malgun Gothic',
    20,
    [System.Drawing.FontStyle]::Bold
)
$headerPanel.Controls.Add($titleLabel)

$subtitleLabel = New-HubLabel -Text '한 번의 클릭으로 실습, 논문 재현, 도구, 자료, 환경 관리를 시작하세요.' -Left 28 -Top 62 -Width 650 -Height 26 -FontSize 10 -ForeColor ([System.Drawing.Color]::FromArgb(224, 233, 248))
$headerPanel.Controls.Add($subtitleLabel)

$environmentBadge = New-HubLabel -Text '환경 상태 확인 중' -Left 815 -Top 24 -Width 175 -Height 34 -FontSize 9
$environmentBadge.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
$environmentBadge.Font = New-Object System.Drawing.Font(
    'Malgun Gothic',
    9,
    [System.Drawing.FontStyle]::Bold
)
$script:EnvironmentBadge = $environmentBadge
$headerPanel.Controls.Add($environmentBadge)

$refreshStatusButton = New-HubButton -Text '상태 새로 고침' -Left 1000 -Top 24 -Width 96 -Height 34 -BackColor ([System.Drawing.Color]::FromArgb(77, 117, 173))
$refreshStatusButton.Font = New-Object System.Drawing.Font('Malgun Gothic', 8)
$refreshStatusButton.Add_Click({
        Set-EnvironmentBadge
    })
$headerPanel.Controls.Add($refreshStatusButton)
Set-EnvironmentBadge

$tabControl = New-Object System.Windows.Forms.TabControl
$tabControl.Location = New-Object System.Drawing.Point(20, 130)
$tabControl.Size = New-Object System.Drawing.Size(1080, 590)
$tabControl.Anchor = (
    [System.Windows.Forms.AnchorStyles]::Top -bor
    [System.Windows.Forms.AnchorStyles]::Bottom -bor
    [System.Windows.Forms.AnchorStyles]::Left -bor
    [System.Windows.Forms.AnchorStyles]::Right
)
$tabControl.Font = New-Object System.Drawing.Font(
    'Malgun Gothic',
    10,
    [System.Drawing.FontStyle]::Bold
)
$form.Controls.Add($tabControl)

$learningTab = New-Object System.Windows.Forms.TabPage
$learningTab.Text = '학습 시작'
$learningTab.BackColor = [System.Drawing.Color]::FromArgb(250, 251, 253)
$tabControl.TabPages.Add($learningTab)

$startHint = New-HubLabel -Text '처음이라면: 환경 관리 탭의 설치/복구 → 전체 환경 점검 → 기본 AI 커리큘럼 00 순서로 시작하세요.' -Left 20 -Top 14 -Width 1020 -Height 24 -FontSize 9 -ForeColor ([System.Drawing.Color]::FromArgb(66, 80, 104))
$learningTab.Controls.Add($startHint)

$basicGroup = New-HubGroup -Title '기본 AI 커리큘럼 (23개)' -Left 20 -Top 45 -Width 515 -Height 180
$learningTab.Controls.Add($basicGroup)
$basicDescription = New-HubLabel -Text '왼쪽에는 TODO 실습본, 오른쪽에는 정답본을 나란히 엽니다.' -Left 18 -Top 31 -Width 470 -Height 24
$basicGroup.Controls.Add($basicDescription)
$basicItems = @(
    '00 | 환경과 JupyterLab',
    '01 | AI를 위한 중급 Python',
    '02 | NumPy for ML',
    '03 | pandas EDA와 그룹 분할',
    '04 | scikit-learn 텍스트 기준선',
    '05 | PyTorch Tensor와 Autograd',
    '06 | PyTorch 텍스트 분류',
    '07 | 신경망 직접 구현',
    '08 | CNN, RNN, Attention 비교',
    '09 | Transformer 직접 구현',
    '10 | Tiny LM 학습과 생성',
    '11 | RAG 검색과 근거화',
    '12 | End-to-end AI 프로젝트',
    '13 | 데이터 모달리티와 파이프라인',
    '14 | RNN 시퀀스 모델링',
    '15 | Transformer 구조 실습',
    '16 | Transformer 학습 실습',
    '17 | 임베딩과 벡터 인덱스',
    '18 | RAG end-to-end',
    '19 | CNN 이미지 분류',
    '20 | 표형 MLP와 이상 탐지',
    '21 | GPU 학습과 최적화',
    '22 | Hybrid RAG 평가'
)
$basicCombo = New-HubComboBox -Items $basicItems -Left 18 -Top 68 -Width 315
$basicGroup.Controls.Add($basicCombo)
$basicButton = New-HubButton -Text '이 실습 열기' -Left 345 -Top 65 -Width 145
$basicButton.Tag = $basicCombo
$basicButton.Add_Click({
        param($sender, $event)
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            $number = Get-ChoiceCode -ComboBox $sender.Tag
            Start-CmdLauncher -LauncherName 'Start_Paired_Learning.cmd' -Arguments @($number)
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$basicGroup.Controls.Add($basicButton)
$basicGuideButton = New-HubButton -Text '읽기 쉬운 전체 가이드' -Left 18 -Top 120 -Width 190 -Height 34 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$basicGuideButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Coding_Practice_Guide.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$basicGroup.Controls.Add($basicGuideButton)

$paperGroup = New-HubGroup -Title '대표 AI 논문 재현 (20편)' -Left 545 -Top 45 -Width 515 -Height 180
$learningTab.Controls.Add($paperGroup)
$paperDescription = New-HubLabel -Text 'LeNet-5부터 CLIP까지, 논문의 핵심을 작게 구현해 봅니다.' -Left 18 -Top 31 -Width 470 -Height 24
$paperGroup.Controls.Add($paperDescription)
$paperItems = @(
    '00 | LeNet-5',
    '01 | AlexNet',
    '02 | U-Net',
    '03 | ResNet',
    '04 | Dropout',
    '05 | Batch Normalization',
    '06 | word2vec Negative Sampling',
    '07 | Sequence to Sequence',
    '08 | Bahdanau Attention',
    '09 | Attention Is All You Need',
    '10 | GPT-1',
    '11 | BERT',
    '12 | VAE',
    '13 | GAN',
    '14 | DQN',
    '15 | GCN',
    '16 | SimCLR',
    '17 | Vision Transformer',
    '18 | DDPM',
    '19 | CLIP'
)
$paperCombo = New-HubComboBox -Items $paperItems -Left 18 -Top 68 -Width 315
$paperGroup.Controls.Add($paperCombo)
$paperButton = New-HubButton -Text '논문 실습 열기' -Left 345 -Top 65 -Width 145
$paperButton.Tag = $paperCombo
$paperButton.Add_Click({
        param($sender, $event)
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            $number = Get-ChoiceCode -ComboBox $sender.Tag
            Start-CmdLauncher -LauncherName 'Start_Paper_Reproductions.cmd' -Arguments @($number)
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$paperGroup.Controls.Add($paperButton)
$paperGuideButton = New-HubButton -Text '논문 학습 지도 보기' -Left 18 -Top 120 -Width 190 -Height 34 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$paperGuideButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Coding_Practice_Guide.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$paperGroup.Controls.Add($paperGuideButton)
$koreanPaperNotesButton = New-HubButton -Text '상세 한국어 논문 해설' -Left 220 -Top 120 -Width 205 -Height 34 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$koreanPaperNotesButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Korean_Paper_Notes.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$paperGroup.Controls.Add($koreanPaperNotesButton)

$fieldGroup = New-HubGroup -Title '분야별 논문 실습 (7개 분야 × 10편)' -Left 20 -Top 240 -Width 515 -Height 190
$learningTab.Controls.Add($fieldGroup)
$fieldDescription = New-HubLabel -Text '관심 분야와 논문 번호를 골라 실습본과 정답본을 함께 엽니다.' -Left 18 -Top 31 -Width 480 -Height 24
$fieldGroup.Controls.Add($fieldDescription)
$fieldItems = @(
    'vision | 컴퓨터 비전',
    'nlp_llm | NLP와 LLM',
    'generative | 생성 모델',
    'reinforcement_learning | 강화학습과 에이전트',
    'graph_recommendation | 그래프와 추천',
    'self_supervised_multimodal | 자기지도와 멀티모달',
    'distillation_compression | 지식 증류와 모델 경량화'
)
$fieldCombo = New-HubComboBox -Items $fieldItems -Left 18 -Top 68 -Width 315
$fieldGroup.Controls.Add($fieldCombo)
$fieldNumberItems = 0..9 | ForEach-Object { '{0:D2} | 분야별 논문' -f $_ }
$fieldNumberCombo = New-HubComboBox -Items $fieldNumberItems -Left 18 -Top 108 -Width 175
$fieldGroup.Controls.Add($fieldNumberCombo)
$fieldButton = New-HubButton -Text '이 실습 열기' -Left 345 -Top 65 -Width 145
$fieldButton.Tag = [pscustomobject]@{
    Field = $fieldCombo
    Number = $fieldNumberCombo
}
$fieldButton.Add_Click({
        param($sender, $event)
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            $fieldId = Get-ChoiceCode -ComboBox $sender.Tag.Field
            $number = Get-ChoiceCode -ComboBox $sender.Tag.Number
            Start-CmdLauncher -LauncherName 'Start_Field_Paper_Labs.cmd' -Arguments @($fieldId, $number)
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$fieldGroup.Controls.Add($fieldButton)
$fieldMenuButton = New-HubButton -Text '분야 선택 메뉴 열기' -Left 205 -Top 107 -Width 190 -Height 34 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$fieldMenuButton.Add_Click({
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            Start-CmdLauncher -LauncherName 'Start_Field_Paper_Labs.cmd'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$fieldGroup.Controls.Add($fieldMenuButton)

$workspaceGroup = New-HubGroup -Title '모든 자료를 둘러보기' -Left 545 -Top 240 -Width 515 -Height 190
$learningTab.Controls.Add($workspaceGroup)
$workspaceDescription = New-HubLabel -Text '일반 JupyterLab으로 전체 노트북을 보거나, 원본 폴더를 바로 엽니다.' -Left 18 -Top 31 -Width 475 -Height 24
$workspaceGroup.Controls.Add($workspaceDescription)
$jupyterButton = New-HubButton -Text 'JupyterLab 열기' -Left 18 -Top 67 -Width 170
$jupyterButton.Add_Click({
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            Start-CmdLauncher -LauncherName 'Start_JupyterLab.cmd'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$workspaceGroup.Controls.Add($jupyterButton)
$notebookButton = New-HubButton -Text '노트북 폴더' -Left 200 -Top 67 -Width 135 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$notebookButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'notebooks'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$workspaceGroup.Controls.Add($notebookButton)
$folderButton = New-HubButton -Text '학습 폴더' -Left 347 -Top 67 -Width 145 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$folderButton.Add_Click({
        try {
            Start-CmdLauncher -LauncherName 'Open_Learning_Folder.cmd'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$workspaceGroup.Controls.Add($folderButton)
$learningPathButton = New-HubButton -Text '브라우저 학습 가이드' -Left 18 -Top 117 -Width 190 -Height 34 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$learningPathButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Coding_Practice_Guide.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$workspaceGroup.Controls.Add($learningPathButton)

$overviewGroup = New-HubGroup -Title '다음에 무엇을 할까요?' -Left 20 -Top 445 -Width 1040 -Height 82
$learningTab.Controls.Add($overviewGroup)
$overviewLabel = New-HubLabel -Text '기초를 다지고 싶다면 기본 00~12, 논문 흐름을 빠르게 훑고 싶다면 대표 논문 00~19, 한 분야를 깊게 파고들고 싶다면 분야별 00~09를 선택하세요.' -Left 18 -Top 31 -Width 990 -Height 30 -FontSize 9 -ForeColor ([System.Drawing.Color]::FromArgb(66, 80, 104))
$overviewGroup.Controls.Add($overviewLabel)

$resourcesTab = New-Object System.Windows.Forms.TabPage
$resourcesTab.Text = '도구와 자료'
$resourcesTab.BackColor = [System.Drawing.Color]::FromArgb(250, 251, 253)
$resourcesTab.AutoScroll = $true
$tabControl.TabPages.Add($resourcesTab)

$appGroup = New-HubGroup -Title '관찰형 실습 도구' -Left 20 -Top 20 -Width 515 -Height 195
$resourcesTab.Controls.Add($appGroup)
$appDescription = New-HubLabel -Text '데이터, 분류 모델, PyTorch tensor, RAG, Tiny LM을 화면에서 관찰합니다.' -Left 18 -Top 31 -Width 475 -Height 42
$appDescription.AutoEllipsis = $false
$appGroup.Controls.Add($appDescription)
$streamlitButton = New-HubButton -Text 'Streamlit 실습 앱 열기' -Left 18 -Top 92 -Width 220
$streamlitButton.Add_Click({
        try {
            Start-StreamlitLab
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$appGroup.Controls.Add($streamlitButton)
$appHint = New-HubLabel -Text '처음 실행하면 브라우저가 자동으로 열립니다.' -Left 18 -Top 142 -Width 330 -Height 24 -FontSize 8.5 -ForeColor ([System.Drawing.Color]::FromArgb(88, 97, 112))
$appGroup.Controls.Add($appHint)

$documentGroup = New-HubGroup -Title '읽기 쉬운 가이드와 자료' -Left 545 -Top 20 -Width 515 -Height 250
$resourcesTab.Controls.Add($documentGroup)
$webGuideButton = New-HubButton -Text '브라우저 학습 가이드' -Left 18 -Top 40 -Width 210 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$webGuideButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Coding_Practice_Guide.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$documentGroup.Controls.Add($webGuideButton)
$paperNotesResourceButton = New-HubButton -Text '상세 한국어 논문 해설' -Left 250 -Top 40 -Width 210 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$paperNotesResourceButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Korean_Paper_Notes.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$documentGroup.Controls.Add($paperNotesResourceButton)
$functionGlossaryButton = New-HubButton -Text '함수·API 한국어 사전' -Left 18 -Top 96 -Width 210 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$functionGlossaryButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Function_Glossary.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$documentGroup.Controls.Add($functionGlossaryButton)
$typingWorkbookButton = New-HubButton -Text 'NLP 타이핑 워크북' -Left 250 -Top 96 -Width 210 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$typingWorkbookButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'projects\nlp\Typing_Practice_Workbook.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$documentGroup.Controls.Add($typingWorkbookButton)
$pdfGuideButton = New-HubButton -Text '인쇄용 PDF 가이드' -Left 18 -Top 152 -Width 210 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$pdfGuideButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'AI_Coding_Practice_Guide.pdf'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$documentGroup.Controls.Add($pdfGuideButton)
$docsFolderButton = New-HubButton -Text '원본 문서 폴더' -Left 250 -Top 152 -Width 210 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$docsFolderButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'docs'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$documentGroup.Controls.Add($docsFolderButton)

$filesGroup = New-HubGroup -Title '프로젝트와 원본 파일' -Left 20 -Top 295 -Width 1040 -Height 160
$resourcesTab.Controls.Add($filesGroup)
$projectDescription = New-HubLabel -Text '노트북 밖의 Python 프로젝트, 실행 예제, 데이터도 여기에서 바로 찾아볼 수 있습니다.' -Left 18 -Top 31 -Width 980 -Height 24
$filesGroup.Controls.Add($projectDescription)
$nlpButton = New-HubButton -Text 'NLP 미니 프로젝트' -Left 18 -Top 73 -Width 180 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$nlpButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'projects\nlp'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$filesGroup.Controls.Add($nlpButton)
$scriptsButton = New-HubButton -Text '실행 예제 스크립트' -Left 210 -Top 73 -Width 190 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$scriptsButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'scripts'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$filesGroup.Controls.Add($scriptsButton)
$dataButton = New-HubButton -Text '로컬 연습 데이터' -Left 412 -Top 73 -Width 180 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$dataButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'data'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$filesGroup.Controls.Add($dataButton)
$toolsButton = New-HubButton -Text '도구 폴더' -Left 604 -Top 73 -Width 160 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$toolsButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'tools'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$filesGroup.Controls.Add($toolsButton)

$typingGroup = New-HubGroup -Title 'NLP 직접 따라치기 - 내 작업본' -Left 20 -Top 470 -Width 1040 -Height 125
$resourcesTab.Controls.Add($typingGroup)
$typingDescription = New-HubLabel -Text '원본 starter.py는 보존하고, learner_work 폴더에 개인 작업본을 한 번만 만들어 직접 타이핑합니다. 정답은 막힐 때 같은 함수만 비교하세요.' -Left 18 -Top 31 -Width 980 -Height 24
$typingGroup.Controls.Add($typingDescription)
$typingItems = @(
    '01 | 텍스트 전처리',
    '02 | 의도 분류',
    '03 | 의미 검색',
    '04 | 프레임워크 없는 RAG',
    '05 | LangChain RAG',
    '06 | RAG 평가'
)
$typingCombo = New-HubComboBox -Items $typingItems -Left 18 -Top 67 -Width 300
$typingGroup.Controls.Add($typingCombo)
$typingOpenButton = New-HubButton -Text '내 starter.py 열기' -Left 332 -Top 64 -Width 190 -Height 36
$typingOpenButton.Tag = $typingCombo
$typingOpenButton.Add_Click({
        param($sender, $event)
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            $projectId = Get-ChoiceCode -ComboBox $sender.Tag
            Start-CmdLauncher -LauncherName 'Start_NLP_Typing_Lab.cmd' -Arguments @($projectId, '--open-editor')
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$typingGroup.Controls.Add($typingOpenButton)
$typingRunButton = New-HubButton -Text '내 코드 실행' -Left 535 -Top 64 -Width 165 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$typingRunButton.Tag = $typingCombo
$typingRunButton.Add_Click({
        param($sender, $event)
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            $projectId = Get-ChoiceCode -ComboBox $sender.Tag
            Start-CmdLauncher -LauncherName 'Start_NLP_Typing_Lab.cmd' -Arguments @($projectId, '--run')
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$typingGroup.Controls.Add($typingRunButton)
$typingGuideButton = New-HubButton -Text '타이핑 워크북 열기' -Left 713 -Top 64 -Width 190 -Height 36 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$typingGuideButton.Add_Click({
        try {
            Open-WorkspaceItem -RelativePath 'projects\nlp\Typing_Practice_Workbook.html'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$typingGroup.Controls.Add($typingGuideButton)

$managementTab = New-Object System.Windows.Forms.TabPage
$managementTab.Text = '환경 관리'
$managementTab.BackColor = [System.Drawing.Color]::FromArgb(250, 251, 253)
$tabControl.TabPages.Add($managementTab)

$setupGroup = New-HubGroup -Title '처음 설치와 GPU 설정' -Left 20 -Top 20 -Width 515 -Height 190
$managementTab.Controls.Add($setupGroup)
$setupDescription = New-HubLabel -Text 'Python 3.12가 준비된 뒤 설치/복구를 실행하세요. 설치에는 처음 한 번 인터넷 연결이 필요합니다.' -Left 18 -Top 31 -Width 475 -Height 44
$setupDescription.AutoEllipsis = $false
$setupGroup.Controls.Add($setupDescription)
$installButton = New-HubButton -Text '설치 또는 복구 실행' -Left 18 -Top 96 -Width 210 -Height 40
$installButton.Add_Click({
        try {
            Start-CmdLauncher -LauncherName 'Install_or_Repair.cmd'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$setupGroup.Controls.Add($installButton)
$gpuButton = New-HubButton -Text 'NVIDIA GPU 가속 설치' -Left 245 -Top 96 -Width 225 -Height 40 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$gpuButton.Add_Click({
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            Start-CmdLauncher -LauncherName 'Install_GPU_PyTorch.cmd'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$setupGroup.Controls.Add($gpuButton)

$verifyGroup = New-HubGroup -Title '환경 점검과 문제 해결' -Left 545 -Top 20 -Width 515 -Height 190
$managementTab.Controls.Add($verifyGroup)
$verifyDescription = New-HubLabel -Text '전체 점검은 테스트, 노트북 구조, 데이터와 실행기를 확인하므로 시간이 걸릴 수 있습니다.' -Left 18 -Top 31 -Width 475 -Height 44
$verifyDescription.AutoEllipsis = $false
$verifyGroup.Controls.Add($verifyDescription)
$verifyButton = New-HubButton -Text '전체 환경 점검 실행' -Left 18 -Top 96 -Width 210 -Height 40
$verifyButton.Add_Click({
        try {
            if (-not (Require-LearningEnvironment)) {
                return
            }
            Start-CmdLauncher -LauncherName 'Verify_Setup.cmd'
        }
        catch {
            Show-HubMessage -Message $_.Exception.Message -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
        }
    })
$verifyGroup.Controls.Add($verifyButton)
$statusButton = New-HubButton -Text '설치 상태 새로 고침' -Left 245 -Top 96 -Width 225 -Height 40 -BackColor ([System.Drawing.Color]::FromArgb(80, 105, 142))
$statusButton.Add_Click({
        Set-EnvironmentBadge
        if (Test-LearningEnvironment) {
            Show-HubMessage -Message '학습 환경을 찾았습니다. 이제 실습과 도구를 실행할 수 있습니다.'
        }
        else {
            Show-HubMessage -Message '아직 .venv가 없습니다. 설치 또는 복구 실행을 먼저 완료하세요.' -Icon ([System.Windows.Forms.MessageBoxIcon]::Warning)
        }
    })
$verifyGroup.Controls.Add($statusButton)

$shortcutGroup = New-HubGroup -Title '바탕화면 단일 아이콘' -Left 20 -Top 235 -Width 1040 -Height 130
$managementTab.Controls.Add($shortcutGroup)
$shortcutText = New-HubLabel -Text '바탕화면의 “AI 코딩 실습 센터” 아이콘 하나가 이 창을 엽니다. 패키지를 다른 폴더로 옮긴 경우에는 이 폴더의 Create_AI_Coding_Lab_Shortcut.ps1을 다시 실행하면 됩니다.' -Left 18 -Top 31 -Width 980 -Height 50 -FontSize 9 -ForeColor ([System.Drawing.Color]::FromArgb(66, 80, 104))
$shortcutText.AutoEllipsis = $false
$shortcutGroup.Controls.Add($shortcutText)

$footerLabel = New-HubLabel -Text '기존 실행 파일과 노트북은 수정하지 않았습니다. 이 센터는 각 기능으로 안전하게 연결해 주는 시작 화면입니다.' -Left 24 -Top 728 -Width 1070 -Height 24 -FontSize 8.5 -ForeColor ([System.Drawing.Color]::FromArgb(92, 101, 115))
$footerLabel.Anchor = (
    [System.Windows.Forms.AnchorStyles]::Bottom -bor
    [System.Windows.Forms.AnchorStyles]::Left -bor
    [System.Windows.Forms.AnchorStyles]::Right
)
$form.Controls.Add($footerLabel)

if ($SmokeTest) {
    $form.CreateControl()
    $form.Dispose()
    Write-Output 'SMOKE: AI 코딩 실습 센터 UI를 만들었습니다.'
    exit 0
}

[void]$form.ShowDialog()
