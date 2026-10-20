param(
    [string]$Revision = "dec3f8f2444eace7004fc084abe1be9f3d88270e"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$target = Join-Path $repoRoot "external\Memora"

if (-not (Test-Path -LiteralPath $target)) {
    git clone https://github.com/microsoft/Memora.git $target
}

git -C $target fetch origin
git -C $target checkout --detach $Revision
$actual = (git -C $target rev-parse HEAD).Trim()
if ($actual -ne $Revision) {
    throw "Memora revision mismatch: expected $Revision, got $actual"
}

Write-Host "Memora ready at $target ($actual)"
