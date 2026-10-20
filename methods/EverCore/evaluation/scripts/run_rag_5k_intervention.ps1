param(
    [ValidateSet("all", "locomo", "longmemeval", "meme")]
    [string]$Dataset = "all",
    [ValidateSet("both", "answer", "evaluate")]
    [string]$Stage = "both",
    [string]$RunName = "5k-unexpand-first-20260920",
    [string]$PythonPath = "python",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$summaryPath = Join-Path $root "evaluation\results\rag_context_cap_$RunName.json"
$summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
$runs = @($summary | Where-Object { $Dataset -eq "all" -or $_.dataset -eq $Dataset })
if ($runs.Count -eq 0) {
    throw "No prepared runs matched $Dataset"
}
if (-not (Get-Command $PythonPath -ErrorAction SilentlyContinue) -and
    -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}
$pythonCommand = (Get-Command $PythonPath -ErrorAction SilentlyContinue).Source
if (-not $pythonCommand) {
    $pythonCommand = $PythonPath
}
$stages = if ($Stage -eq "both") { @("answer", "evaluate") } else { @($Stage) }
$envNames = @("LLM_MODEL", "EVALUATE_MODEL", "MEME_EVALUATE_MODEL", "MEME_JUDGE_NUM_RUNS", "PYTHONUTF8", "PYTHONIOENCODING")
$savedEnv = @{}
foreach ($name in $envNames) {
    $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
Push-Location $root
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    foreach ($run in $runs) {
        $outputDir = Join-Path $root ("evaluation\results\" + $run.output_dir)
        $searchPath = Join-Path $outputDir "search_results.json"
        $hash = (Get-FileHash -LiteralPath $searchPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $run.output_search_sha256) {
            throw "Prepared search results changed: $searchPath"
        }
        $env:LLM_MODEL = [string]$run.answer_model
        $env:EVALUATE_MODEL = [string]$run.judge_model
        $env:MEME_EVALUATE_MODEL = [string]$run.judge_model
        $env:MEME_JUDGE_NUM_RUNS = [string]$run.judge_num_runs
        $cliArgs = @(
            "-m", "evaluation.cli",
            "--dataset", [string]$run.dataset,
            "--system", "rag",
            "--run-name", $RunName,
            "--output-dir", $outputDir,
            "--stages"
        ) + $stages
        Write-Host "$($run.dataset): reader=$($run.answer_model), judge=$($run.judge_model), judge runs=$($run.judge_num_runs)"
        Write-Host "$PythonPath $($cliArgs -join ' ')"
        if (-not $DryRun) {
            & $pythonCommand @cliArgs
            if ($LASTEXITCODE -ne 0) {
                throw "$($run.dataset) failed with exit code $LASTEXITCODE"
            }
        }
    }
}
finally {
    Pop-Location
    foreach ($name in $envNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], "Process")
    }
}
