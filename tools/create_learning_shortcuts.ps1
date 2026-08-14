<#
.SYNOPSIS
Creates the learning-launcher shortcuts used by this workspace.

.DESCRIPTION
By default, shortcuts are created on the current user's Desktop. Existing .lnk
files with the same names are updated in place; no other Desktop entries are
enumerated, moved, or removed.

.PARAMETER OutputDirectory
Optional destination used for testing or for a non-Desktop installation. A
relative path is resolved from the workspace root.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File tools\create_learning_shortcuts.ps1 -WhatIf

.EXAMPLE
powershell -ExecutionPolicy Bypass -File tools\create_learning_shortcuts.ps1 `
    -OutputDirectory artifacts\shortcut_qa
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
param(
    [Parameter()]
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $destinationDirectory = [Environment]::GetFolderPath('Desktop')
    if ([string]::IsNullOrWhiteSpace($destinationDirectory)) {
        throw 'Windows Desktop 폴더를 찾지 못했습니다.'
    }
}
elseif ([IO.Path]::IsPathRooted($OutputDirectory)) {
    $destinationDirectory = [IO.Path]::GetFullPath($OutputDirectory)
}
else {
    $destinationDirectory = [IO.Path]::GetFullPath(
        (Join-Path $workspaceRoot $OutputDirectory)
    )
}

$pairedLauncher = Join-Path $workspaceRoot 'Start_Paired_Learning.cmd'
$paperLauncher = Join-Path $workspaceRoot 'Start_Paper_Reproductions.cmd'
$fieldLauncher = Join-Path $workspaceRoot 'Start_Field_Paper_Labs.cmd'
$readingNotes = Join-Path $workspaceRoot 'docs\paper_reading_notes\README.md'
$commandProcessor = (Get-Command 'cmd.exe' -ErrorAction Stop).Source
$explorerTarget = (Get-Command 'explorer.exe' -ErrorAction Stop).Source

function Get-LauncherArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Launcher,

        [Parameter()]
        [string]$LauncherArguments = ''
    )

    # Pointing a .lnk directly at a .cmd file under a Unicode/OneDrive path can
    # produce a shortcut whose TargetPath is blank. Use cmd.exe as the stable
    # executable and quote the launcher invocation as cmd /s /c requires.
    $invocation = '"' + $Launcher + '"'
    if (-not [string]::IsNullOrWhiteSpace($LauncherArguments)) {
        $invocation += ' ' + $LauncherArguments
    }
    return '/d /s /c "' + $invocation + '"'
}

foreach ($requiredLauncher in @($pairedLauncher, $paperLauncher, $fieldLauncher)) {
    if (-not (Test-Path -LiteralPath $requiredLauncher -PathType Leaf)) {
        throw "필수 실행 파일이 없습니다: $requiredLauncher"
    }
}

$shortcuts = @(
    [pscustomobject]@{
        Name = 'AI 코딩연습 - 기본 실습과 정답'
        Target = $commandProcessor
        Arguments = (Get-LauncherArguments -Launcher $pairedLauncher)
        Description = 'AI 코딩연습 기본 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 20편'
        Target = $commandProcessor
        Arguments = (Get-LauncherArguments -Launcher $paperLauncher)
        Description = 'AI 주요 논문 20편의 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - 전체 분야'
        Target = $commandProcessor
        Arguments = (Get-LauncherArguments -Launcher $fieldLauncher)
        Description = 'AI 분야별 논문 실습 선택 메뉴를 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - 컴퓨터 비전'
        Target = $commandProcessor
        Arguments = (Get-LauncherArguments -Launcher $fieldLauncher -LauncherArguments 'vision')
        Description = '컴퓨터 비전 분야 논문 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - NLP·LLM'
        Target = $commandProcessor
        Arguments = (Get-LauncherArguments -Launcher $fieldLauncher -LauncherArguments 'nlp_llm')
        Description = 'NLP·LLM 분야 논문 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - 생성 모델'
        Target = $commandProcessor
        Arguments = (Get-LauncherArguments -Launcher $fieldLauncher -LauncherArguments 'generative')
        Description = '생성 모델 분야 논문 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - 강화학습·에이전트'
        Target = $commandProcessor
        Arguments = (
            Get-LauncherArguments `
                -Launcher $fieldLauncher `
                -LauncherArguments 'reinforcement_learning'
        )
        Description = '강화학습·에이전트 분야 논문 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - 그래프·추천'
        Target = $commandProcessor
        Arguments = (
            Get-LauncherArguments `
                -Launcher $fieldLauncher `
                -LauncherArguments 'graph_recommendation'
        )
        Description = '그래프·추천 분야 논문 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - 자기지도·멀티모달'
        Target = $commandProcessor
        Arguments = (
            Get-LauncherArguments `
                -Launcher $fieldLauncher `
                -LauncherArguments 'self_supervised_multimodal'
        )
        Description = '자기지도·멀티모달 분야 논문 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 실습 - 지식 증류·모델 경량화'
        Target = $commandProcessor
        Arguments = (
            Get-LauncherArguments `
                -Launcher $fieldLauncher `
                -LauncherArguments 'distillation_compression'
        )
        Description = '지식 증류·모델 경량화 분야 논문 실습본과 정답본을 함께 엽니다.'
    }
    [pscustomobject]@{
        Name = 'AI 논문 한국어 요약'
        Target = $explorerTarget
        Arguments = '"' + $readingNotes + '"'
        Description = 'AI 논문 한국어 독해 노트의 목차를 엽니다.'
    }
)

if (-not (Test-Path -LiteralPath $destinationDirectory -PathType Container)) {
    if ($PSCmdlet.ShouldProcess($destinationDirectory, '바로가기 출력 폴더 만들기')) {
        $null = New-Item -ItemType Directory -Path $destinationDirectory -Force
    }
}

$wscriptShell = $null
$savedCount = 0
try {
    foreach ($definition in $shortcuts) {
        $shortcutPath = Join-Path $destinationDirectory ($definition.Name + '.lnk')
        if (-not $PSCmdlet.ShouldProcess($shortcutPath, '바로가기 만들기 또는 업데이트')) {
            continue
        }

        if ($null -eq $wscriptShell) {
            $wscriptShell = New-Object -ComObject 'WScript.Shell'
        }

        $shortcut = $wscriptShell.CreateShortcut($shortcutPath)
        try {
            $shortcut.TargetPath = $definition.Target
            $shortcut.Arguments = $definition.Arguments
            $shortcut.WorkingDirectory = $workspaceRoot
            $shortcut.Description = $definition.Description
            $shortcut.WindowStyle = 1
            $shortcut.Save()
            $savedCount += 1
            Write-Output "CREATED $shortcutPath"
        }
        finally {
            $isComShortcut = (
                $null -ne $shortcut -and
                [Runtime.InteropServices.Marshal]::IsComObject($shortcut)
            )
            if ($isComShortcut) {
                $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut)
            }
        }
    }
}
finally {
    $isComShell = (
        $null -ne $wscriptShell -and
        [Runtime.InteropServices.Marshal]::IsComObject($wscriptShell)
    )
    if ($isComShell) {
        $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($wscriptShell)
    }
}

if ($WhatIfPreference) {
    Write-Output "WHATIF 완료: 바로가기 $($shortcuts.Count)개의 생성·업데이트 계획을 확인했습니다."
}
else {
    Write-Output "PASS: 바로가기 ${savedCount}개를 만들거나 업데이트했습니다: $destinationDirectory"
}
