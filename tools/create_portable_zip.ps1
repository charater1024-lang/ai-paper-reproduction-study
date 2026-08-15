[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$OutputDirectory = "artifacts\releases",
    [string]$ArchiveName = "",
    [switch]$IncludeNotebookOutputs
)

$ErrorActionPreference = "Stop"

function Get-FullPathInsideRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $rootFull = [System.IO.Path]::GetFullPath($Root)
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        [System.IO.Path]::GetFullPath($Path)
    }
    else {
        [System.IO.Path]::GetFullPath((Join-Path $rootFull $Path))
    }
    $rootPrefix = $rootFull.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith(
        $rootPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label must stay inside $rootFull. Resolved path: $candidate"
    }
    return $candidate
}

function Test-ZipEntryPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EntryName
    )

    $normalized = $EntryName.Replace("\", "/")
    if ($normalized.StartsWith("/") -or $normalized -match "^[A-Za-z]:") {
        return $false
    }
    foreach ($segment in $normalized.Split("/")) {
        if ($segment -eq "..") {
            return $false
        }
    }
    return $true
}

$repositoryRoot = (
    Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
).Path
$artifactsRoot = Get-FullPathInsideRoot `
    -Root $repositoryRoot `
    -Path "artifacts" `
    -Label "Artifacts directory"
$outputRoot = Get-FullPathInsideRoot `
    -Root $repositoryRoot `
    -Path $OutputDirectory `
    -Label "Output directory"
$artifactsPrefix = $artifactsRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$outputIsInsideArtifacts = $outputRoot.StartsWith(
    $artifactsPrefix,
    [System.StringComparison]::OrdinalIgnoreCase
) -or $outputRoot.Equals(
    $artifactsRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)
if (-not $outputIsInsideArtifacts) {
    throw "Output directory must stay inside $artifactsRoot. Resolved: $outputRoot"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to collect the portable project file list."
}

& git -C $repositoryRoot rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) {
    throw "The workspace is not a Git working tree: $repositoryRoot"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($ArchiveName)) {
    $ArchiveName = "ai-paper-reproduction-study-windows-portable-$timestamp.zip"
}
if ([System.IO.Path]::GetFileName($ArchiveName) -ne $ArchiveName) {
    throw "ArchiveName must be a file name without directory components."
}
if (-not $ArchiveName.EndsWith(".zip", [System.StringComparison]::OrdinalIgnoreCase)) {
    $ArchiveName += ".zip"
}
$archivePath = Join-Path $outputRoot $ArchiveName
if (Test-Path -LiteralPath $archivePath) {
    throw "Archive already exists. Choose another name: $archivePath"
}

$candidateFiles = @(
    & git -C $repositoryRoot ls-files --cached --others --exclude-standard
)
if ($LASTEXITCODE -ne 0 -or $candidateFiles.Count -eq 0) {
    throw "Git did not return a portable project file list."
}
$candidateFiles = @($candidateFiles | Sort-Object -Unique)

$nonRegularTrackedFiles = @(
    & git -C $repositoryRoot ls-files --stage |
        Where-Object { $_ -match '^(120000|160000) ' }
)
if ($LASTEXITCODE -ne 0) {
    throw "Git could not inspect tracked file modes."
}
if ($nonRegularTrackedFiles.Count -gt 0) {
    throw "Refusing to package symlinks or Git submodules."
}

$blockedPathPatterns = @(
    '(^|/)(\.env[^/]*|[^/]+\.env(?:\.[^/]*)?|\.envrc)(/|$)',
    '(^|/)\.streamlit/secrets\.toml$',
    '(^|/)(id_rsa|id_dsa|id_ecdsa|id_ed25519)(\..*)?$',
    '\.(pem|p12|pfx|key|ppk|jks|keystore|keytab|ovpn)(\..*)?$',
    '(^|/)(credentials|service-account|secrets?)(\.[^/]*)?$',
    '(^|/)(\.npmrc|\.pypirc|\.netrc|\.git-credentials)$',
    '(^|/)\.aws/credentials$',
    '(^|/)\.docker/config\.json$',
    '(^|/)\.kube/config$',
    '(^|/)terraform\.tfstate(?:\..*)?$',
    '(^|/)terraform\.tfvars$',
    '\.auto\.tfvars$'
)
foreach ($relativePath in $candidateFiles) {
    $normalizedPath = $relativePath.Replace("\", "/")
    foreach ($pattern in $blockedPathPatterns) {
        if ($normalizedPath -match $pattern) {
            throw "Refusing to package a possible secret file: $relativePath"
        }
    }
}

if (-not $PSCmdlet.ShouldProcess(
    $archivePath,
    "Create a portable ZIP from $($candidateFiles.Count) project files"
)) {
    return
}

New-Item -ItemType Directory -Path $artifactsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stagingName = "portable-staging-" + [System.Guid]::NewGuid().ToString("N")
$stagingRoot = Get-FullPathInsideRoot `
    -Root $artifactsRoot `
    -Path $stagingName `
    -Label "Staging directory"
New-Item -ItemType Directory -Path $stagingRoot | Out-Null

$archiveCreated = $false
$archiveVerified = $false
try {
    $copiedFiles = 0
    $copiedBytes = [int64]0
    $copiedRelativePaths = @()
    foreach ($relativePath in $candidateFiles) {
        $sourcePath = Get-FullPathInsideRoot `
            -Root $repositoryRoot `
            -Path $relativePath `
            -Label "Project file"
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            throw "Selected project file is missing or not a regular file: $relativePath"
        }
        $sourceItem = Get-Item -LiteralPath $sourcePath
        if ($sourceItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to package a reparse-point file: $relativePath"
        }
        $destinationPath = Get-FullPathInsideRoot `
            -Root $stagingRoot `
            -Path $relativePath `
            -Label "Package file"
        $destinationDirectory = Split-Path -Parent $destinationPath
        New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath
        $copiedFiles += 1
        $copiedBytes += (Get-Item -LiteralPath $sourcePath).Length
        $copiedRelativePaths += $relativePath.Replace("\", "/")
    }
    if ($copiedFiles -ne $candidateFiles.Count) {
        throw "Copied file count does not match the Git-selected file count."
    }

    $clearedOutputCells = 0
    if (-not $IncludeNotebookOutputs) {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        $notebookFiles = Get-ChildItem -LiteralPath $stagingRoot `
            -Filter "*.ipynb" `
            -File `
            -Recurse
        foreach ($notebookFile in $notebookFiles) {
            $notebookText = [System.IO.File]::ReadAllText($notebookFile.FullName)
            $notebook = $notebookText | ConvertFrom-Json
            $notebookChanged = $false
            foreach ($cell in $notebook.cells) {
                if ($cell.cell_type -ne "code") {
                    continue
                }
                $hasOutputs = $null -ne $cell.outputs -and $cell.outputs.Count -gt 0
                $hasExecutionCount = $null -ne $cell.execution_count
                if ($hasOutputs -or $hasExecutionCount) {
                    $clearedOutputCells += 1
                    $notebookChanged = $true
                }
                $cell.outputs = @()
                $cell.execution_count = $null
            }
            if ($notebookChanged) {
                $sanitizedJson = $notebook | ConvertTo-Json -Depth 100
                [System.IO.File]::WriteAllText(
                    $notebookFile.FullName,
                    $sanitizedJson + [Environment]::NewLine,
                    $utf8NoBom
                )
            }
        }
    }

    $packagedPayloadBytes = [int64]0
    foreach ($packagedFile in Get-ChildItem -LiteralPath $stagingRoot -File -Recurse) {
        $packagedPayloadBytes += $packagedFile.Length
    }

    $branch = (& git -C $repositoryRoot branch --show-current).Trim()
    $commit = (& git -C $repositoryRoot rev-parse HEAD).Trim()
    $dirtyCount = @(& git -C $repositoryRoot status --porcelain=v1).Count
    $packageInfo = @(
        "# Portable package information",
        "",
        "- Created: $(Get-Date -Format o)",
        "- Source branch: $branch",
        "- Source commit: $commit",
        "- Working-tree changes included: $dirtyCount",
        "- Project payload files: $copiedFiles",
        "- Source project payload bytes before notebook cleanup: $copiedBytes",
        "- Packaged project payload bytes after notebook cleanup: $packagedPayloadBytes",
        "- Notebook code cells with saved runtime state removed: $clearedOutputCells",
        "",
        "## Start on another Windows PC",
        "",
        "1. Prepare 64-bit Python 3.12 and an internet connection for first install.",
        "2. Extract the ZIP to a short final path such as C:\ai-paper-lab.",
        "3. Run Install_or_Repair.cmd.",
        "4. If an NVIDIA GPU is available, run Install_GPU_PyTorch.cmd.",
        "5. Run Verify_Setup.cmd.",
        "6. Recreate shortcuts with tools\create_learning_shortcuts.ps1.",
        "7. Start with Start_Paired_Learning.cmd 00.",
        "",
        "See README.md and docs/WINDOWS_SETUP.md for the full guide.",
        "The ZIP intentionally excludes .git, .venv, caches, secrets, and runtime artifacts.",
        "Learner code is preserved; saved outputs and execution counters are removed by default.",
        "If the extracted folder moves, rerun the installer and shortcut creator."
    )
    $packageInfoPath = Join-Path $stagingRoot "PORTABLE_PACKAGE_INFO.md"
    $packageInfo | Set-Content -LiteralPath $packageInfoPath -Encoding UTF8

    $hashLines = @()
    $payloadFiles = Get-ChildItem -LiteralPath $stagingRoot -File -Recurse |
        Sort-Object FullName
    foreach ($file in $payloadFiles) {
        $relative = $file.FullName.Substring($stagingRoot.Length + 1).Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()
        $hashLines += "$hash  $relative"
    }
    $hashManifestPath = Join-Path $stagingRoot "SHA256SUMS.txt"
    $hashLines | Set-Content -LiteralPath $hashManifestPath -Encoding UTF8

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $stagingRoot,
        $archivePath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $false
    )
    $archiveCreated = $true

    $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $manifestEntry = $archive.GetEntry("SHA256SUMS.txt")
        if ($null -eq $manifestEntry) {
            throw "SHA256SUMS.txt is missing from the archive."
        }
        $manifestReader = New-Object System.IO.StreamReader($manifestEntry.Open())
        try {
            $manifestText = $manifestReader.ReadToEnd()
        }
        finally {
            $manifestReader.Dispose()
        }
        $expectedHashes = @{}
        foreach ($line in $manifestText -split "`r?`n") {
            $cleanLine = $line.TrimStart([char]0xFEFF)
            if ([string]::IsNullOrWhiteSpace($cleanLine)) {
                continue
            }
            if ($cleanLine -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
                throw "Invalid SHA256SUMS.txt line: $cleanLine"
            }
            $expectedHashes[$Matches[2]] = $Matches[1].ToLowerInvariant()
        }

        $expectedEntries = @{}
        foreach ($relativePath in $copiedRelativePaths) {
            $expectedEntries[$relativePath] = $true
        }
        $expectedEntries["PORTABLE_PACKAGE_INFO.md"] = $true
        $expectedEntries["SHA256SUMS.txt"] = $true

        $entryNames = @{}
        $notebookCount = 0
        $rootNotebookCount = 0
        $pairedExerciseCount = 0
        $pairedSolutionCount = 0
        $paperExerciseCount = 0
        $paperSolutionCount = 0
        $fieldExerciseCount = 0
        $fieldSolutionCount = 0
        $verifiedPayloadCount = 0
        foreach ($entry in $archive.Entries) {
            if (-not (Test-ZipEntryPath -EntryName $entry.FullName)) {
                throw "Unsafe ZIP entry path: $($entry.FullName)"
            }
            if ([string]::IsNullOrEmpty($entry.Name)) {
                continue
            }
            $entryName = $entry.FullName.Replace("\", "/")
            $caseFoldedName = $entryName.ToLowerInvariant()
            if ($entryNames.ContainsKey($caseFoldedName)) {
                throw "Case-insensitive duplicate ZIP entry: $entryName"
            }
            $entryNames[$caseFoldedName] = $true
            if (-not $expectedEntries.ContainsKey($entryName)) {
                throw "Unexpected ZIP entry: $entryName"
            }
            if (
                $entryName -match '(^|/)(\.git|\.venv|venv|__pycache__|\.ipynb_checkpoints)(/|$)' -or
                $entryName -match '\.(pyc|pyo|pt|pth|ckpt|log)$'
            ) {
                throw "Forbidden non-portable ZIP entry: $entryName"
            }
            if ($entryName.EndsWith(".ipynb")) {
                $notebookCount += 1
                switch -Regex ($entryName) {
                    '^notebooks/[0-9]{2}_[^/]+\.ipynb$' {
                        $rootNotebookCount += 1
                        break
                    }
                    '^notebooks/exercises/[0-9]{2}_[^/]+\.ipynb$' {
                        $pairedExerciseCount += 1
                        break
                    }
                    '^notebooks/solutions/[0-9]{2}_[^/]+\.ipynb$' {
                        $pairedSolutionCount += 1
                        break
                    }
                    '^notebooks/paper_reproductions/exercises/[0-9]{2}_[^/]+\.ipynb$' {
                        $paperExerciseCount += 1
                        break
                    }
                    '^notebooks/paper_reproductions/solutions/[0-9]{2}_[^/]+\.ipynb$' {
                        $paperSolutionCount += 1
                        break
                    }
                    '^notebooks/field_reproductions/[^/]+/exercises/[0-9]{2}_[^/]+\.ipynb$' {
                        $fieldExerciseCount += 1
                        break
                    }
                    '^notebooks/field_reproductions/[^/]+/solutions/[0-9]{2}_[^/]+\.ipynb$' {
                        $fieldSolutionCount += 1
                        break
                    }
                    default {
                        throw "Unexpected notebook path in ZIP: $entryName"
                    }
                }
            }
            if ($entryName -eq "SHA256SUMS.txt") {
                continue
            }
            if (-not $expectedHashes.ContainsKey($entryName)) {
                throw "Missing expected hash for ZIP entry: $entryName"
            }
            $stream = $entry.Open()
            try {
                $sha256 = [System.Security.Cryptography.SHA256]::Create()
                try {
                    $hashBytes = $sha256.ComputeHash($stream)
                }
                finally {
                    $sha256.Dispose()
                }
            }
            finally {
                $stream.Dispose()
            }
            $actualHash = -join ($hashBytes | ForEach-Object { $_.ToString("x2") })
            if ($actualHash -ne $expectedHashes[$entryName]) {
                throw "SHA-256 mismatch for ZIP entry: $entryName"
            }
            $verifiedPayloadCount += 1
        }
        if ($entryNames.Count -ne $expectedEntries.Count) {
            throw "ZIP entry count does not match the selected project file list."
        }
        if ($verifiedPayloadCount -ne $expectedHashes.Count) {
            throw "SHA-256 manifest count does not match verified ZIP payloads."
        }
        if ($notebookCount -ne 239) {
            throw "Expected 239 canonical notebooks, found $notebookCount in ZIP."
        }
        $expectedNotebookCounts = @(
            @("root originals", $rootNotebookCount, 13),
            @("paired exercises", $pairedExerciseCount, 23),
            @("paired solutions", $pairedSolutionCount, 23),
            @("paper exercises", $paperExerciseCount, 20),
            @("paper solutions", $paperSolutionCount, 20),
            @("field exercises", $fieldExerciseCount, 70),
            @("field solutions", $fieldSolutionCount, 70)
        )
        foreach ($countContract in $expectedNotebookCounts) {
            if ($countContract[1] -ne $countContract[2]) {
                throw (
                    "Expected {0} {1}, found {2}." -f
                    $countContract[2],
                    $countContract[0],
                    $countContract[1]
                )
            }
        }
        $entryCount = $archive.Entries.Count
    }
    finally {
        $archive.Dispose()
    }

    $archiveItem = Get-Item -LiteralPath $archivePath
    $archiveHash = (
        Get-FileHash -LiteralPath $archivePath -Algorithm SHA256
    ).Hash.ToLower()
    $archiveVerified = $true
    [pscustomobject]@{
        Archive = $archiveItem.FullName
        SizeBytes = $archiveItem.Length
        Entries = $entryCount
        SHA256 = $archiveHash
    } | Format-List
}
finally {
    if ($archiveCreated -and -not $archiveVerified -and (Test-Path -LiteralPath $archivePath)) {
        $resolvedArchive = [System.IO.Path]::GetFullPath($archivePath)
        $outputPrefix = $outputRoot.TrimEnd(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        ) + [System.IO.Path]::DirectorySeparatorChar
        $safeArchive = $resolvedArchive.StartsWith(
            $outputPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -and (
            (Split-Path -Leaf $resolvedArchive) -eq $ArchiveName
        )
        if (-not $safeArchive) {
            throw "Refusing to remove unexpected archive path: $resolvedArchive"
        }
        Remove-Item -LiteralPath $resolvedArchive -Force
    }
    if (Test-Path -LiteralPath $stagingRoot) {
        $resolvedStaging = [System.IO.Path]::GetFullPath($stagingRoot)
        $artifactsPrefix = $artifactsRoot.TrimEnd(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        ) + [System.IO.Path]::DirectorySeparatorChar
        $safeToRemove = $resolvedStaging.StartsWith(
            $artifactsPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -and (
            (Split-Path -Leaf $resolvedStaging).StartsWith("portable-staging-")
        )
        if (-not $safeToRemove) {
            throw "Refusing to remove unexpected staging path: $resolvedStaging"
        }
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
}
