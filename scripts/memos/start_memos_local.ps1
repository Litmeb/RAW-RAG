param(
    [string]$RepoDir = "external/MemOS",
    [string]$SourceEnv = "methods/EverCore/.env",
    [string]$GitUrl = "https://github.com/MemTensor/MemOS.git",
    [string]$GitRef = "",
    [switch]$Build,
    [switch]$Pull
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $scriptDir = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $scriptDir "../..")).Path
}

function Read-DotEnv {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $values
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0 -or $trimmed.StartsWith("#")) {
            continue
        }
        $name, $value = $trimmed -split "=", 2
        if (-not $name -or $null -eq $value) {
            continue
        }
        $values[$name.Trim()] = $value.Trim().Trim('"').Trim("'")
    }
    return $values
}

function Get-EnvValue {
    param(
        [hashtable]$Values,
        [string]$Name,
        [string]$Default = ""
    )

    $processValue = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ($processValue) {
        return $processValue
    }
    if ($Values.ContainsKey($Name) -and $Values[$Name]) {
        return $Values[$Name]
    }
    return $Default
}

$root = Get-RepoRoot
$repoPath = Join-Path $root $RepoDir
$sourceEnvPath = Join-Path $root $SourceEnv

if (-not (Test-Path -LiteralPath $repoPath)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $repoPath) -Force | Out-Null
    git clone --depth 1 $GitUrl $repoPath
}

if ($GitRef) {
    git -C $repoPath fetch --depth 1 origin $GitRef
    git -C $repoPath checkout FETCH_HEAD
} elseif ($Pull) {
    git -C $repoPath pull --ff-only
}

$envValues = Read-DotEnv -Path $sourceEnvPath

$llmApiKey = Get-EnvValue -Values $envValues -Name "MEMOS_OPENAI_API_KEY" -Default (Get-EnvValue -Values $envValues -Name "LLM_API_KEY")
$llmBaseUrl = Get-EnvValue -Values $envValues -Name "MEMOS_OPENAI_API_BASE" -Default (Get-EnvValue -Values $envValues -Name "LLM_BASE_URL" -Default "https://api.openai.com/v1")
$llmModel = Get-EnvValue -Values $envValues -Name "MEMOS_CHAT_MODEL" -Default (Get-EnvValue -Values $envValues -Name "LLM_MODEL" -Default "gpt-4o-mini")
$readerModel = Get-EnvValue -Values $envValues -Name "MEMOS_MEMREADER_MODEL" -Default $llmModel
$readerApiKey = Get-EnvValue -Values $envValues -Name "MEMOS_MEMREADER_API_KEY" -Default $llmApiKey
$readerBaseUrl = Get-EnvValue -Values $envValues -Name "MEMOS_MEMREADER_API_BASE" -Default $llmBaseUrl
$embeddingModel = Get-EnvValue -Values $envValues -Name "MEMOS_EMBEDDER_MODEL" -Default (Get-EnvValue -Values $envValues -Name "VECTORIZE_MODEL" -Default "text-embedding-3-small")
$embeddingApiKey = Get-EnvValue -Values $envValues -Name "MEMOS_EMBEDDER_API_KEY" -Default (Get-EnvValue -Values $envValues -Name "VECTORIZE_API_KEY" -Default $llmApiKey)
$embeddingBaseUrl = Get-EnvValue -Values $envValues -Name "MEMOS_EMBEDDER_API_BASE" -Default (Get-EnvValue -Values $envValues -Name "VECTORIZE_BASE_URL" -Default $llmBaseUrl)
$embeddingDimension = Get-EnvValue -Values $envValues -Name "MEMOS_EMBEDDING_DIMENSION" -Default "1536"

if (-not $llmApiKey) {
    throw "No LLM API key found. Set LLM_API_KEY in $SourceEnv or MEMOS_OPENAI_API_KEY in the current shell."
}
if (-not $embeddingApiKey) {
    throw "No embedding API key found. Set VECTORIZE_API_KEY in $SourceEnv or MEMOS_EMBEDDER_API_KEY in the current shell."
}

$memosEnvPath = Join-Path $repoPath ".env"
$memosEnv = @(
    "OPENAI_API_KEY=$llmApiKey",
    "OPENAI_API_BASE=$llmBaseUrl",
    "MOS_CHAT_MODEL=$llmModel",
    "MEMRADER_MODEL=$readerModel",
    "MEMRADER_API_KEY=$readerApiKey",
    "MEMRADER_API_BASE=$readerBaseUrl",
    "MEMREADER_GENERAL_API_KEY=$readerApiKey",
    "MEMREADER_GENERAL_API_BASE=$readerBaseUrl",
    "PREFERENCE_EXTRACTOR_API_KEY=$readerApiKey",
    "PREFERENCE_EXTRACTOR_API_BASE=$readerBaseUrl",
    "MOS_EMBEDDER_BACKEND=universal_api",
    "MOS_EMBEDDER_MODEL=$embeddingModel",
    "MOS_EMBEDDER_API_KEY=$embeddingApiKey",
    "MOS_EMBEDDER_API_BASE=$embeddingBaseUrl",
    "EMBEDDING_DIMENSION=$embeddingDimension",
    "MOS_RERANKER_BACKEND=cosine_local",
    "NEO4J_BACKEND=neo4j-community",
    "NEO4J_URI=bolt://neo4j-docker:7687",
    "NEO4J_USER=neo4j",
    "NEO4J_PASSWORD=$(Get-EnvValue -Values $envValues -Name 'NEO4J_PASSWORD' -Default 'change-me-local-only')",
    "NEO4J_DB_NAME=neo4j",
    "MOS_NEO4J_SHARED_DB=false",
    "DEFAULT_USE_REDIS_QUEUE=false",
    "ENABLE_CHAT_API=false",
    "FILE_LOCAL_PATH=/tmp",
    "MOS_MAX_TOKENS=8000",
    "MEMRADER_MAX_TOKENS=8000",
    "MEMREADER_GENERAL_MAX_TOKENS=8000",
    "PREFERENCE_EXTRACTOR_MAX_TOKENS=8000"
)
Set-Content -LiteralPath $memosEnvPath -Value $memosEnv -Encoding UTF8

$neo4jImage = Get-EnvValue -Values $envValues -Name "MEMOS_NEO4J_IMAGE" -Default "neo4j:5.26.6"
$neo4jImageId = docker images -q $neo4jImage
if (-not $neo4jImageId) {
    $fallbackNeo4jImageId = docker images -q "neo4j:5.26.4"
    if ($fallbackNeo4jImageId) {
        $neo4jImage = "neo4j:5.26.4"
    }
}
$env:MEMOS_NEO4J_IMAGE = $neo4jImage
$env:MEMOS_QDRANT_IMAGE = Get-EnvValue -Values $envValues -Name "MEMOS_QDRANT_IMAGE" -Default "qdrant/qdrant:v1.15.3"

$overrideCompose = Join-Path $root "scripts/memos/docker-compose.local.yml"
$composeArgs = @(
    "compose",
    "-f",
    (Join-Path $repoPath "docker/docker-compose.yml"),
    "-f",
    $overrideCompose,
    "up",
    "-d"
)
if ($Build) {
    $composeArgs += "--build"
}

docker @composeArgs
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "MemOS local Docker is starting at http://localhost:8000"
Write-Host "Health check: Invoke-RestMethod http://localhost:8000/health"
Write-Host "Evaluation URL: MEMOS_LOCAL_URL=http://localhost:8000"
