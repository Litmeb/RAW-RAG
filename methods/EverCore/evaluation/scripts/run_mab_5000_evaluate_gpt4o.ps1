param(
    [string]$JudgeModel = "gpt-4o-2024-11-20",
    [string]$SummaryPath = "evaluation/results/memoryagentbench_context_cap_5000_token-budget-5000-min1-best-20260713.json",
    [string]$PythonPath = "python",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$everCoreRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $everCoreRoot

if (-not (Get-Command $PythonPath -ErrorAction SilentlyContinue) -and
    -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}
$pythonCommand = (Get-Command $PythonPath -ErrorAction SilentlyContinue).Source
if (-not $pythonCommand) {
    $pythonCommand = $PythonPath
}

$resolvedSummary = (Resolve-Path -LiteralPath $SummaryPath).Path
$summary = Get-Content -Raw -LiteralPath $resolvedSummary | ConvertFrom-Json
$runs = @($summary | Where-Object { $_.needs_answer_evaluate -eq $true })
if ($runs.Count -ne 44) {
    throw "Expected 44 capped source runs, found $($runs.Count): $resolvedSummary"
}

$env:MEMORYAGENTBENCH_EVALUATE_MODEL = $JudgeModel
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "MemoryAgentBench evaluate-only batch"
Write-Host "Judge model: $JudgeModel"
Write-Host "Runs: $($runs.Count)"
Write-Host "Summary: $resolvedSummary"

$failures = @()
$index = 0
foreach ($run in $runs) {
    $index += 1
    $source = [string]$run.source
    if ($source -eq "longmemeval_s_") {
        $source = "longmemeval_s*"
    }

    $outputDir = Join-Path "evaluation/results" ([string]$run.output_dir)
    $answerPath = Join-Path $outputDir "answer_results.json"
    if (-not (Test-Path -LiteralPath $answerPath -PathType Leaf)) {
        throw "Missing answer result: $answerPath"
    }

    $cliArgs = @(
        "-m", "evaluation.cli",
        "--dataset", "memoryagentbench",
        "--system", [string]$run.system,
        "--mab-split", [string]$run.split,
        "--mab-source", $source,
        "--run-name", "token-budget-5000-min1-best-20260713",
        "--output-dir", $outputDir,
        "--stages", "evaluate"
    )

    Write-Host ""
    Write-Host "[$index/$($runs.Count)] $($run.system) | $($run.split) | $source"
    Write-Host "$PythonPath $($cliArgs -join ' ')"
    if ($DryRun) {
        continue
    }

    & $pythonCommand @cliArgs
    if ($LASTEXITCODE -ne 0) {
        $failures += "$($run.system)/$($run.split)/$source (exit=$LASTEXITCODE)"
        Write-Warning "Evaluate stage failed; continuing with the remaining runs."
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete; no judge API calls were made."
    exit 0
}
if ($failures.Count -gt 0) {
    Write-Error ("Evaluate batch completed with failures:`n- " + ($failures -join "`n- "))
    exit 1
}

Write-Host "Evaluate batch completed successfully for all $($runs.Count) runs."
