param(
    [switch]$SkipPreprocessing
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Invoke-Step {
    param(
        [string]$Label,
        [string]$ScriptPath
    )

    Write-Host ""
    Write-Host "=== $Label ==="
    & python $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipPreprocessing) {
    Invoke-Step "Step 1: Preprocessing" (Join-Path $root "01_preprocessing.py")
}

Invoke-Step "Step 2: Case Representation" (Join-Path $root "python 02_case_representation.py")
Invoke-Step "Step 3: Retrieval" (Join-Path $root "python 03_retrieval.py")
Invoke-Step "Step 4: Prediction" (Join-Path $root "python 04_predict.py")
Invoke-Step "Step 5: Evaluation" (Join-Path $root "python 05_evaluation.py")

Write-Host ""
Write-Host "Pipeline completed successfully."
