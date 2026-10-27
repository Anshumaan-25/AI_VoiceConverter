param([string]$PythonExe = '', [switch]$SkipAssets)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe" }
}
if (-not (Test-Path -LiteralPath 'venv-rvc/Scripts/python.exe')) {
    if ($PythonExe) { Invoke-Checked $PythonExe @('-m','venv','venv-rvc') }
    else { Invoke-Checked 'py' @('-3.10','-m','venv','venv-rvc') }
}
$RvcPython = Join-Path $PSScriptRoot 'venv-rvc/Scripts/python.exe'
Invoke-Checked $RvcPython @('-c','import importlib.util,shutil,sys; assert sys.version_info[:2] == (3,10); budget = 1 if importlib.util.find_spec("torch") else 8; assert shutil.disk_usage(".").free >= (10+budget)*1024**3, "Insufficient estimated install space plus 10 GiB reserve"')
Invoke-Checked $RvcPython @('-m','pip','install','pip==24.0','setuptools==80.9.0','wheel==0.45.1')
Invoke-Checked $RvcPython @('-m','pip','install','torch==2.1.0+cu121','torchaudio==2.1.0+cu121','--index-url','https://download.pytorch.org/whl/cu121')
Invoke-Checked $RvcPython @('-m','pip','install','-r','requirements-rvc.txt','-c','requirements-rvc-lock.txt')
Invoke-Checked $RvcPython @('-B','restore_rvc.py')
if (-not $SkipAssets) { Invoke-Checked $RvcPython @('-B','download_rvc_assets.py') }
Invoke-Checked $RvcPython @('-B','rvc_runtime.py','--health')
