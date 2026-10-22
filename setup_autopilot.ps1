param([string]$PythonExe = "", [string]$VenvDir = "venv310")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $Arguments" }
}
if (-not (Test-Path -LiteralPath (Join-Path $VenvDir 'Scripts/python.exe'))) {
    if ($PythonExe) { Invoke-Checked $PythonExe @('-m', 'venv', $VenvDir) }
    else { Invoke-Checked 'py' @('-3.10', '-m', 'venv', $VenvDir) }
}
$Vpy = Join-Path $VenvDir 'Scripts/python.exe'
Invoke-Checked $Vpy @('-c', 'import sys; assert sys.version_info[:2] == (3,10), "Python 3.10 required"')
Invoke-Checked $Vpy @('-m', 'pip', 'install', 'pip==24.0', 'setuptools==80.9.0', 'wheel==0.45.1')
Invoke-Checked $Vpy @('-m', 'pip', 'install', 'torch==2.1.0+cu121', 'torchaudio==2.1.0+cu121', '--index-url', 'https://download.pytorch.org/whl/cu121')
Invoke-Checked $Vpy @('-m', 'pip', 'install', '-r', 'requirements-runtime.txt', '-c', 'requirements-lock.txt')
Invoke-Checked $Vpy @('restore_svc.py')
Invoke-Checked $Vpy @('health_check.py')
