# Setup team collaboration directories and git config for Windows
try {
    Write-Host "Configuring git..."
    git config --local core.protectNTFS false
    Write-Host "Git core.protectNTFS set to false"
} catch {
    Write-Warning "Unable to run git config. Is git installed and repository initialized? $_"
}

$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path "$base\.."
$workDir = Join-Path $repoRoot "team_work"
$bestModels = Join-Path $repoRoot "best_models"

New-Item -ItemType Directory -Path $workDir -Force | Out-Null
New-Item -ItemType Directory -Path $bestModels -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $bestModels ".gitkeep") -Force | Out-Null

$members = @("Y_Fu", "Y_Yao", "X_Jiang")
foreach ($m in $members) {
    $dir = Join-Path $workDir $m
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "Created $dir"
}

Write-Host "\n=== Collaboration Rules ==="
Write-Host "1) Commit code + scripts only. Do NOT commit full training checkpoints or dataset dumps."
Write-Host "2) Use scripts/manage_models.py to register best models to best_models/ with metadata."
Write-Host "3) Keep a clean working tree: pull before start and push at end."
Write-Host "4) Use standard branches (e.g., feature/<name>, fix/<name>) and PR review."
Write-Host "5) Maintain README_TEAM.md with workflow and notes."
Write-Host "6) Move heavy artifacts to shared storage if not in best_models."
Write-Host "========================\n"
