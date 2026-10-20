param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$Python = "D:/Anaconda/envs/py312pt291cu128/python.exe",
    [string]$RunName = "memos-local-smoke"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $PSCommandPath
$root = (Resolve-Path (Join-Path $scriptDir "../..")).Path
$evalDir = Join-Path $root "methods/EverCore"

$health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
if ($health.status -ne "healthy") {
    throw "MemOS health check failed: $($health | ConvertTo-Json -Compress)"
}

Push-Location $evalDir
try {
    $env:PYTHONPATH = "src"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    $env:LLM_API_KEY = if ($env:LLM_API_KEY) { $env:LLM_API_KEY } else { "dummy" }
    $env:MEMOS_LOCAL_URL = $BaseUrl

    & $Python -c @"
from pathlib import Path
from evaluation.src.utils.config import load_yaml
from evaluation.src.adapters.registry import create_adapter

cfg = load_yaml('evaluation/config/systems/memos_local.yaml')
cfg['dataset_name'] = 'locomo'
cfg['run_name'] = '$RunName'
adapter = create_adapter(
    cfg['adapter'],
    cfg,
    output_dir=Path('evaluation/results/_memos_local_product_adapter_smoke'),
)
print(adapter.api_url)
print(adapter.api_flavor)
print(adapter._add_url())
print(adapter._search_url())
"@
}
finally {
    Pop-Location
}
