$ErrorActionPreference = "Stop"

$pretrainDir = Join-Path $PSScriptRoot "so-vits-svc/pretrain"
New-Item -ItemType Directory -Force -Path $pretrainDir | Out-Null

$hubertOut = Join-Path $pretrainDir "checkpoint_best_legacy_500.pt"
$rmvpeOut = Join-Path $pretrainDir "rmvpe.pt"

if ((Test-Path -LiteralPath $hubertOut) -and (Test-Path -LiteralPath $rmvpeOut)) {
    Write-Host "Existing weights preserved. Move them aside explicitly to replace them."
    exit 0
}
$tmpDir = Join-Path $pretrainDir ("_download_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmpDir | Out-Null
try {
    if (-not (Test-Path -LiteralPath $hubertOut)) {
        Write-Host "Downloading ContentVec checkpoint"
        $hubertStage = Join-Path $tmpDir "hubert.pt"
        Invoke-WebRequest -Uri "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" -OutFile $hubertStage
        if ((Get-Item -LiteralPath $hubertStage).Length -lt 1000000) { throw "Invalid encoder download" }
        Move-Item -LiteralPath $hubertStage -Destination $hubertOut
    }
    if (-not (Test-Path -LiteralPath $rmvpeOut)) {
        Write-Host "Downloading RMVPE archive"
        $rmvpeZip = Join-Path $tmpDir "rmvpe.zip"
        Invoke-WebRequest -Uri "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip" -OutFile $rmvpeZip
        Expand-Archive -LiteralPath $rmvpeZip -DestinationPath $tmpDir
        $rmvpeStage = Join-Path $tmpDir "model.pt"
        if ((Get-Item -LiteralPath $rmvpeStage).Length -lt 1000000) { throw "Invalid RMVPE download" }
        Move-Item -LiteralPath $rmvpeStage -Destination $rmvpeOut
    }
}
finally {
    $resolvedRoot = [IO.Path]::GetFullPath($pretrainDir) + [IO.Path]::DirectorySeparatorChar
    $resolvedTemp = [IO.Path]::GetFullPath($tmpDir)
    if (-not $resolvedTemp.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing cleanup outside pretrained directory"
    }
    Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
}

Write-Host "Done. Files created:"
Get-Item $hubertOut, $rmvpeOut | Select-Object FullName, Length
