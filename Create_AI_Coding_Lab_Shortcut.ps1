#requires -Version 5.1
<#
.SYNOPSIS
Creates one Desktop shortcut for the AI Coding Practice Center.

.DESCRIPTION
The shortcut targets powershell.exe rather than the .cmd wrapper so the
launcher window can stay hidden while the Windows Forms hub is shown.
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
param(
    [Parameter()]
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$hubScript = Join-Path $workspaceRoot 'AI_Coding_Practice_Center.ps1'

if (-not (Test-Path -LiteralPath $hubScript -PathType Leaf)) {
    throw "학습 센터 스크립트를 찾지 못했습니다: $hubScript"
}

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

if (-not (Test-Path -LiteralPath $destinationDirectory -PathType Container)) {
    if ($PSCmdlet.ShouldProcess($destinationDirectory, '바로가기 출력 폴더 만들기')) {
        $null = New-Item -ItemType Directory -Path $destinationDirectory -Force
    }
}

$shortcutPath = Join-Path $destinationDirectory 'AI 코딩 실습 센터.lnk'
if (-not $PSCmdlet.ShouldProcess($shortcutPath, '바탕화면 단일 바로가기 만들기 또는 업데이트')) {
    return
}

$powerShellPath = (Get-Command 'powershell.exe' -ErrorAction Stop).Source
$wscriptShell = $null
$shortcut = $null

try {
    $wscriptShell = New-Object -ComObject 'WScript.Shell'
    $shortcut = $wscriptShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $powerShellPath
    $shortcut.Arguments = (
        '-NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' +
        $hubScript +
        '"'
    )
    $shortcut.WorkingDirectory = $workspaceRoot
    $shortcut.Description = 'AI 코딩 실습, 논문 재현, 도구와 환경 관리를 한 곳에서 엽니다.'
    $shortcut.IconLocation = (
        (Join-Path $env:SystemRoot 'System32\imageres.dll') +
        ',102'
    )
    $shortcut.WindowStyle = 7
    $shortcut.Save()
}
finally {
    if ($null -ne $shortcut -and [Runtime.InteropServices.Marshal]::IsComObject($shortcut)) {
        $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut)
    }
    if ($null -ne $wscriptShell -and [Runtime.InteropServices.Marshal]::IsComObject($wscriptShell)) {
        $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($wscriptShell)
    }
}

Write-Output "CREATED $shortcutPath"
