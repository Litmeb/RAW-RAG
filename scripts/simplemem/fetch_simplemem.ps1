param([string]$Revision = 'db80b6a7c591e0ea730a058e9f5fc4eb06572299')

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$target = Join-Path $repoRoot 'external\SimpleMem'
if (-not (Test-Path -LiteralPath $target)) {
    git clone --depth 1 https://github.com/aiming-lab/SimpleMem.git $target
    if ($LASTEXITCODE -ne 0) { throw 'SimpleMem clone failed' }
}
$changes = git -C $target status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect SimpleMem checkout' }
if ($changes) { throw 'SimpleMem checkout has local changes; preserve them before switching revision' }
$actual = git -C $target rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read SimpleMem revision' }
if ($actual.Trim() -ne $Revision) {
    git -C $target fetch --depth 1 origin $Revision
    if ($LASTEXITCODE -ne 0) { throw 'SimpleMem fetch failed' }
    git -C $target checkout --detach $Revision
    if ($LASTEXITCODE -ne 0) { throw 'SimpleMem checkout failed' }
}
Write-Host "SimpleMem ready at $target ($Revision)"
