#Requires -Version 7.0
[CmdletBinding()]
param(
    [string[]]$Condition = @(),
    [ValidateSet("Main", "Appendix", "All")]
    [string]$Group = "Main",
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$Tag = "paper",
    [string]$Python = "python",
    [switch]$List,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$packageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../.."))
$evercore = Join-Path $packageRoot "methods/EverCore"
$manifest = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot "conditions.json") |
    ConvertFrom-Json
$ids = @($Condition | ForEach-Object { $_ -split "," } | Where-Object { $_ })
foreach ($id in $ids) {
    if ($id -notin $manifest.conditions.id) { throw "Unknown condition: $id" }
}
$selected = @($manifest.conditions | Where-Object {
    if ($ids.Count) { $_.id -in $ids }
    else { $Group -eq "All" -or $_.group -eq $Group }
})
if ($List) {
    $selected | Select-Object id, group, mode, source_condition, expected_questions |
        Format-Table -AutoSize
    return
}

function Invoke-PaperPython {
    param([string[]]$Arguments, [string]$SaveOutput = "")
    Write-Host ("python " + (($Arguments | ForEach-Object { "'$_'" }) -join " "))
    if ($DryRun) { return }
    if ($SaveOutput) {
        $output = & $Python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Python failed (exit $LASTEXITCODE)." }
        $output | Set-Content -LiteralPath $SaveOutput -Encoding utf8
        $output | Write-Output
    }
    else {
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Python failed (exit $LASTEXITCODE)." }
    }
}

function Get-ResultPath {
    param($Spec, [string]$ConditionId, $Source)
    $prefix = "$($Spec.dataset)-rag"
    if ($null -ne $Source) {
        # Match evaluation.cli.sanitize_path_component for the explicit source list.
        $sourceSlug = $Source.source -replace '[^A-Za-z0-9._-]', '_'
        $prefix += "-$($Source.split)-$sourceSlug"
    }
    return Join-Path $evercore "evaluation/results/$prefix-$Tag-$ConditionId"
}

Write-Host "PYTHONHASHSEED=$($manifest.python_hash_seed)"
if (-not $DryRun) {
    Set-Location -LiteralPath $evercore
    $env:PYTHONHASHSEED = [string]$manifest.python_hash_seed
}
foreach ($spec in $selected) {
    Write-Host "`n=== $($spec.id) [$($spec.mode)] ==="
    $systemSource = Join-Path $PSScriptRoot "configs/systems/$($spec.system_config).yaml"
    $datasetSource = Join-Path $PSScriptRoot "configs/datasets/$($spec.dataset_config).yaml"
    $promptSource = Join-Path $PSScriptRoot "configs/prompts.yaml"
    $snapshot = Join-Path $evercore "evaluation/results/paper-configs/$Tag-$($spec.id)"
    Write-Host "Copy $systemSource -> evaluation/config/systems/rag.yaml"
    Write-Host "Copy $datasetSource -> evaluation/config/datasets/$($spec.dataset).yaml"
    Write-Host "Copy $promptSource -> evaluation/config/prompts.yaml"
    if (-not $DryRun) {
        Copy-Item -LiteralPath $systemSource -Destination "$evercore/evaluation/config/systems/rag.yaml" -Force
        Copy-Item -LiteralPath $datasetSource -Destination "$evercore/evaluation/config/datasets/$($spec.dataset).yaml" -Force
        Copy-Item -LiteralPath $promptSource -Destination "$evercore/evaluation/config/prompts.yaml" -Force
        New-Item -ItemType Directory -Path $snapshot -Force | Out-Null
        Copy-Item -LiteralPath $systemSource -Destination "$snapshot/system.yaml" -Force
        Copy-Item -LiteralPath $datasetSource -Destination "$snapshot/dataset.yaml" -Force
        Copy-Item -LiteralPath $promptSource -Destination "$snapshot/prompts.yaml" -Force
        @{
            condition = $spec
            tag = $Tag
            python = $Python
            python_hash_seed = $manifest.python_hash_seed
            system_sha256 = (Get-FileHash -LiteralPath $systemSource).Hash
            dataset_sha256 = (Get-FileHash -LiteralPath $datasetSource).Hash
            prompts_sha256 = (Get-FileHash -LiteralPath $promptSource).Hash
        } | ConvertTo-Json -Depth 12 |
            Set-Content -LiteralPath "$snapshot/condition.json" -Encoding utf8
    }
    if ($spec.dataset -eq "meme") {
        Write-Host "Set MEME_JUDGE_NUM_RUNS from the selected dataset YAML."
        if (-not $DryRun) {
            # The CLI otherwise lets a stale process/.env value override this YAML.
            $runs = & $Python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['evaluation']['num_runs'])" $datasetSource
            if ($LASTEXITCODE -ne 0) { throw "Cannot read MEME judge repetitions." }
            $env:MEME_JUDGE_NUM_RUNS = ([string]$runs).Trim()
        }
    }
    if ($spec.mode -eq "noabs") {
        $sourceRun = Get-ResultPath $spec $spec.source_condition $null
        Invoke-PaperPython -Arguments @(
            "-m", "evaluation.scripts.analyze_longmemeval_no_abs",
            "--run-dir", $sourceRun, "--json"
        ) -SaveOutput "$snapshot/non_abstention.json"
        continue
    }
    $sources = @($null)
    if ($spec.dataset -eq "memoryagentbench") { $sources = @($manifest.mab_sources) }
    foreach ($source in $sources) {
        $outputDir = Get-ResultPath $spec $spec.id $source
        $stages = @("add", "search", "answer", "evaluate")
        if ($spec.mode -eq "answer_only") {
            $sourceRun = Get-ResultPath $spec $spec.source_condition $source
            Write-Host "Reuse $sourceRun/search_results.json -> $outputDir/search_results.json"
            if (-not $DryRun) {
                New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
                Copy-Item -LiteralPath "$sourceRun/search_results.json" -Destination "$outputDir/search_results.json" -Force
            }
            $stages = @("answer", "evaluate")
        }
        $cliArgs = @(
            "-m", "evaluation.cli", "--dataset", $spec.dataset, "--system", "rag",
            "--run-name", "$Tag-$($spec.id)", "--output-dir", $outputDir
        )
        if ($null -ne $source) {
            $cliArgs += @("--mab-split", $source.split, "--mab-source", $source.source)
        }
        if ($spec.dataset -eq "longmemeval") { $cliArgs += @("--lme-judge", "evermemos") }
        $cliArgs += @("--stages") + $stages
        Invoke-PaperPython -Arguments $cliArgs
        if ($spec.id -like "locomo-*-minimal") {
            Invoke-PaperPython -Arguments @(
                "-m", "evaluation.scripts.score_locomo_lexical",
                "--run-dir", $outputDir, "--with-judge",
                "--expected-questions", [string]$spec.expected_questions,
                "--expected-judgments", "3"
            )
        }
    }
    if ($spec.dataset -eq "memoryagentbench") {
        Invoke-PaperPython -Arguments @(
            "-m", "evaluation.cli", "--dataset", "memoryagentbench", "--system", "rag",
            "--run-name", "$Tag-$($spec.id)", "--mab-report-accuracy"
        ) -SaveOutput "$snapshot/mab_accuracy.txt"
    }
}
