param(
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$PythonArgs = @(
    'scripts\extract_assembly_stance_rows.py',
    '--source', 'trash_dataset',
    '--output-dir', 'outputs\assembly_stance\full_15_22',
    '--finalize'
)

if ($Foreground) {
    Push-Location $Root
    try {
        & python @PythonArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

$LogDir = Join-Path $Root 'outputs\assembly_stance\full_15_22'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stdout = Join-Path $LogDir 'launcher_stdout.log'
$Stderr = Join-Path $LogDir 'launcher_stderr.log'
$Process = Start-Process -FilePath 'python' -ArgumentList $PythonArgs -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
try {
    $Process.PriorityClass = 'BelowNormal'
}
catch {
    # The extractor itself remains correct if the host rejects priority changes.
}
Write-Output "Started resumable Assembly stance extraction. PID=$($Process.Id)"
Write-Output "Progress: $LogDir\extract.log"
Write-Output "Resume after interruption: powershell -ExecutionPolicy Bypass -File scripts\run_assembly_stance_resumable.ps1"
