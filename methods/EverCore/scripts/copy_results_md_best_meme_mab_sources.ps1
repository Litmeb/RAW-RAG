param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path,
    [string]$SourceResultsRoot,
    [string]$ArchiveRoot,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not $SourceResultsRoot) {
    $SourceResultsRoot = Join-Path $RepoRoot "methods\EverCore\evaluation\results"
}
if (-not $ArchiveRoot) {
    $ArchiveRoot = Join-Path $RepoRoot "methods\EverCore\evaluation\results_llm_sweep_sources"
}

$manifest = Join-Path $PSScriptRoot "results_md_best_meme_mab_runs.tsv"
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "Manifest not found: $manifest"
}

New-Item -ItemType Directory -Force -Path $ArchiveRoot | Out-Null

$copied = 0
$skipped = 0
$missing = @()
$totalBytes = 0L

Get-Content -LiteralPath $manifest | ForEach-Object {
    $line = $_
    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
        return
    }

    $cols = $line -split "`t"
    if ($cols.Count -lt 6 -or $cols[0] -ne "1") {
        return
    }

    $runName = $cols[5]
    $source = Join-Path $SourceResultsRoot $runName
    $target = Join-Path $ArchiveRoot $runName

    if (-not (Test-Path -LiteralPath $source)) {
        $missing += $runName
        Write-Warning "Missing source run: $runName"
        return
    }

    $bytes = (Get-ChildItem -LiteralPath $source -Recurse -File | Measure-Object -Property Length -Sum).Sum
    if ($bytes) { $totalBytes += [int64]$bytes }

    if (Test-Path -LiteralPath $target) {
        if ($Force) {
            Remove-Item -LiteralPath $target -Recurse -Force
        } else {
            $skipped += 1
            Write-Host "Skip existing: $runName"
            return
        }
    }

    Write-Host "Copy: $runName"
    Copy-Item -LiteralPath $source -Destination $target -Recurse
    $copied += 1
}

Write-Host ""
Write-Host "Archive root: $ArchiveRoot"
Write-Host ("Copied: {0}, skipped existing: {1}, missing: {2}" -f $copied, $skipped, $missing.Count)
Write-Host ("Selected source size: {0:N2} GB" -f ($totalBytes / 1GB))

if ($missing.Count -gt 0) {
    Write-Host "Missing runs:"
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}
