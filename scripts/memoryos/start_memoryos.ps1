param(
    [string]$ContainerName = "memoryos-memoryagentbench",
    [int]$Port = 8010,
    [string]$Image = "ghcr.io/bai-lab/memoryos:latest",
    [string]$EnvFile = ""
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$sourceEnv = if ($EnvFile) {
    (Resolve-Path $EnvFile).Path
} else {
    Join-Path $repoRoot "methods/EverCore/.env"
}

if (-not (Test-Path -LiteralPath $sourceEnv -PathType Leaf)) {
    throw "Environment file not found: $sourceEnv"
}

function Get-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^\s*$([regex]::Escape($Name))\s*=\s*(.*?)\s*$") {
            $value = $Matches[1].Trim()
            if ($value.Length -ge 2) {
                $first = $value[0]
                $last = $value[$value.Length - 1]
                if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            return $value
        }
    }
    return ""
}

$llmModel = Get-DotEnvValue -Path $sourceEnv -Name "LLM_MODEL"
$llmApiKey = Get-DotEnvValue -Path $sourceEnv -Name "LLM_API_KEY"
$llmBaseUrl = Get-DotEnvValue -Path $sourceEnv -Name "LLM_BASE_URL"
$vectorizeModel = Get-DotEnvValue -Path $sourceEnv -Name "VECTORIZE_MODEL"
$vectorizeApiKey = Get-DotEnvValue -Path $sourceEnv -Name "VECTORIZE_API_KEY"
$vectorizeBaseUrl = Get-DotEnvValue -Path $sourceEnv -Name "VECTORIZE_BASE_URL"

$dataPath = Join-Path $repoRoot "methods/EverCore/evaluation/data/memoryos"
New-Item -ItemType Directory -Force -Path $dataPath | Out-Null

docker pull $Image
docker rm -f $ContainerName 2>$null
docker run -d --name $ContainerName `
    --gpus all `
    -p "${Port}:8010" `
    -e "LLM_MODEL=$llmModel" `
    -e "LLM_API_KEY=$llmApiKey" `
    -e "LLM_BASE_URL=$llmBaseUrl" `
    -e "VECTORIZE_MODEL=$vectorizeModel" `
    -e "VECTORIZE_API_KEY=$vectorizeApiKey" `
    -e "VECTORIZE_BASE_URL=$vectorizeBaseUrl" `
    -e "OPENAI_API_KEY=$llmApiKey" `
    -e "OPENAI_BASE_URL=$llmBaseUrl" `
    -e "MEMORYOS_LLM_MODEL=$llmModel" `
    -e "MEMORYOS_EMBEDDING_MODEL=$vectorizeModel" `
    -e "MEMORYOS_USE_GPU=true" `
    -e "MEMORYOS_DATA_DIR=/app/memoryos_data" `
    -v "${repoRoot}/scripts/memoryos/memoryos_http_server.py:/app/memoryos_http_server.py:ro" `
    -v "${dataPath}:/app/memoryos_data" `
    $Image python /app/memoryos_http_server.py

Write-Host "MemoryOS bridge: http://localhost:$Port/health"
