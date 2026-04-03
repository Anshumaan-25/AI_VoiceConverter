param(
    [string]$PythonExe = "C:/Users/anshu/AppData/Local/Programs/Python/Python310/python.exe",
    [string]$VenvDir = ".venv310"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Creating virtual environment at $VenvDir"
& $PythonExe -m venv $VenvDir

$Vpy = Join-Path $VenvDir "Scripts/python.exe"
$Vpip = Join-Path $VenvDir "Scripts/pip.exe"

Write-Host "[2/5] Upgrading pip/setuptools/wheel"
& $Vpy -m pip install --upgrade pip setuptools wheel

# fairseq/omegaconf metadata compatibility with modern pip requires pip < 24.1
Write-Host "[2b/5] Pinning pip to 24.0 for fairseq compatibility"
& $Vpy -m pip install "pip<24.1"

# librosa 0.9.x still imports pkg_resources; keep setuptools below 81.
Write-Host "[2c/5] Pinning setuptools for librosa compatibility"
& $Vpy -m pip install "setuptools<81"

Write-Host "[3/5] Installing PyTorch CUDA 12.1 wheels"
& $Vpip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "[4/5] Installing so-vits requirements"
& $Vpip install -r "so-vits-svc/requirements.txt"

Write-Host "[5/5] Installing app utilities"
& $Vpip install gradio soundfile imageio-ffmpeg

Write-Host "Setup complete. Activate with: .\$VenvDir\Scripts\Activate.ps1"
Write-Host "Then run env check: .\$VenvDir\Scripts\python.exe voice_autopilot.py check"
