$ErrorActionPreference = "Stop"

$pretrainDir = "so-vits-svc/pretrain"
New-Item -ItemType Directory -Force -Path $pretrainDir | Out-Null

$hubertOut = Join-Path $pretrainDir "checkpoint_best_legacy_500.pt"
$rmvpeZip = Join-Path $pretrainDir "rmvpe.zip"
$rmvpeOut = Join-Path $pretrainDir "rmvpe.pt"

Write-Host "[1/3] Downloading ContentVec checkpoint"
Invoke-WebRequest -Uri "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" -OutFile $hubertOut

Write-Host "[2/3] Downloading RMVPE archive"
Invoke-WebRequest -Uri "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip" -OutFile $rmvpeZip

Write-Host "[3/3] Extracting rmvpe.pt"
$tmpDir = Join-Path $pretrainDir "_rmvpe_tmp"
Expand-Archive -Path $rmvpeZip -DestinationPath $tmpDir -Force
Copy-Item -Path (Join-Path $tmpDir "model.pt") -Destination $rmvpeOut -Force
Remove-Item -Recurse -Force $tmpDir
Remove-Item -Force $rmvpeZip

Write-Host "Done. Files created:"
Get-Item $hubertOut, $rmvpeOut | Select-Object FullName, Length
