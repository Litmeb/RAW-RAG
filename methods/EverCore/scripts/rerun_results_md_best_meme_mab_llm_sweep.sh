#!/usr/bin/env bash
set -uo pipefail

# Batch rerun the 6-system MEME / MemoryAgentBench runs referenced by
# methods/results.md with alternate answer and judge LLMs.
#
# Git Bash example:
#   cd methods/EverCore
#   ANSWER_MODELS="deepseek-v4-flash" \
#   MEME_JUDGE_MODELS="gpt-4o-2024-11-20" \
#   MAB_JUDGE_MODELS="gpt-4o-2024-11-20" \
#   bash scripts/rerun_results_md_best_meme_mab_llm_sweep.sh
#
# EMem answer-only reruns are supported when the target directory keeps
# search_results.json copied from the source run.
# The script runs answer and evaluate as separate Python processes so answer
# can use DeepSeek credentials while judge uses GPT-4o credentials.
# MEME judge defaults to one judging run. Override with MEME_JUDGE_NUM_RUNS.
#
# By default, the script records the active sweep tag in RESULTS_ROOT so a
# rerun after failure/interruption reuses existing target directories. The state
# file is removed after a fully successful sweep. Override behavior with
# SWEEP_TAG, RESUME_EXISTING=0, or STOP_ON_ERROR=0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVERCORE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVERCORE_DIR/../.." && pwd)"

MANIFEST="${MANIFEST:-$SCRIPT_DIR/results_md_best_meme_mab_runs.tsv}"
SOURCE_ROOT="${SOURCE_ROOT:-$EVERCORE_DIR/evaluation/results_llm_sweep_sources}"
RESULTS_ROOT="${RESULTS_ROOT:-$EVERCORE_DIR/evaluation/results}"
ENV_FILE="${ENV_FILE:-$EVERCORE_DIR/.env}"
SWEEP_TAG="${SWEEP_TAG:-}"
SWEEP_TAG_STATE_FILE="${SWEEP_TAG_STATE_FILE:-$RESULTS_ROOT/.results_md_best_meme_mab_llm_sweep_tag}"
SWEEP_TAG_MANAGED_STATE=0

ANSWER_MODELS="${ANSWER_MODELS:-deepseek-v4-flash}"
MEME_JUDGE_MODELS="${MEME_JUDGE_MODELS:-gpt-4o-2024-11-20}"
MAB_JUDGE_MODELS="${MAB_JUDGE_MODELS:-gpt-4o-2024-11-20}"
MEME_JUDGE_NUM_RUNS="${MEME_JUDGE_NUM_RUNS:-1}"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-D:/Anaconda/Scripts/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-py312pt291cu128}"

ANSWER_API_KEY_ENV="${ANSWER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
ANSWER_BASE_URL_ENV="${ANSWER_BASE_URL_ENV:-DEEPSEEK_BASE_URL}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-LLM_API_KEY}"
JUDGE_BASE_URL_ENV="${JUDGE_BASE_URL_ENV:-LLM_BASE_URL}"

ANSWER_LLM_API_KEY="${ANSWER_LLM_API_KEY:-}"
ANSWER_LLM_BASE_URL="${ANSWER_LLM_BASE_URL:-}"
JUDGE_LLM_API_KEY="${JUDGE_LLM_API_KEY:-}"
JUDGE_LLM_BASE_URL="${JUDGE_LLM_BASE_URL:-}"

IFS=',' read -r -a ANSWER_MODEL_LIST <<< "$ANSWER_MODELS"
IFS=',' read -r -a MEME_JUDGE_MODEL_LIST <<< "$MEME_JUDGE_MODELS"
IFS=',' read -r -a MAB_JUDGE_MODEL_LIST <<< "$MAB_JUDGE_MODELS"

sanitize_component() {
    printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//; s/_$//'
}

load_env_file() {
    local env_file="$1"
    [[ -f "$env_file" ]] || return 0

    local line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" == export\ * ]] && line="${line#export }"
        [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue

        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        value="${value%%#*}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi

        if [[ -z "${!key+x}" ]]; then
            export "$key=$value"
        fi
    done < "$env_file"
}

env_value() {
    local name="$1"
    printf '%s' "${!name:-}"
}

initialize_sweep_tag() {
    if [[ -n "$SWEEP_TAG" ]]; then
        echo "Using explicit sweep tag: $SWEEP_TAG"
        return 0
    fi

    if [[ -f "$SWEEP_TAG_STATE_FILE" ]]; then
        IFS= read -r SWEEP_TAG < "$SWEEP_TAG_STATE_FILE" || true
        SWEEP_TAG="${SWEEP_TAG%$'\r'}"
        if [[ -n "$SWEEP_TAG" ]]; then
            if [[ "$DRY_RUN" != "1" ]]; then
                SWEEP_TAG_MANAGED_STATE=1
            fi
            echo "Resuming sweep tag from state file: $SWEEP_TAG"
            return 0
        fi
    fi

    SWEEP_TAG="llm_sweep_$(date +%Y%m%d_%H%M%S)"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "Dry run sweep tag: $SWEEP_TAG"
        return 0
    fi

    mkdir -p "$(dirname "$SWEEP_TAG_STATE_FILE")"
    printf '%s\n' "$SWEEP_TAG" > "$SWEEP_TAG_STATE_FILE"
    SWEEP_TAG_MANAGED_STATE=1
    echo "Starting new sweep tag: $SWEEP_TAG"
}

clear_sweep_tag_state() {
    if [[ "$SWEEP_TAG_MANAGED_STATE" == "1" ]]; then
        rm -f "$SWEEP_TAG_STATE_FILE"
    fi
}

resolve_llm_endpoints() {
    ANSWER_LLM_API_KEY="${ANSWER_LLM_API_KEY:-$(env_value "$ANSWER_API_KEY_ENV")}"
    ANSWER_LLM_BASE_URL="${ANSWER_LLM_BASE_URL:-$(env_value "$ANSWER_BASE_URL_ENV")}"
    JUDGE_LLM_API_KEY="${JUDGE_LLM_API_KEY:-$(env_value "$JUDGE_API_KEY_ENV")}"
    JUDGE_LLM_BASE_URL="${JUDGE_LLM_BASE_URL:-$(env_value "$JUDGE_BASE_URL_ENV")}"

    ANSWER_LLM_BASE_URL="${ANSWER_LLM_BASE_URL:-https://api.deepseek.com/v1}"
    JUDGE_LLM_BASE_URL="${JUDGE_LLM_BASE_URL:-https://www.dmxapi.cn/v1}"
}

validate_llm_endpoints() {
    local missing=0
    if [[ -z "$ANSWER_LLM_API_KEY" ]]; then
        echo "Missing answer API key. Set ANSWER_LLM_API_KEY or $ANSWER_API_KEY_ENV." >&2
        missing=1
    fi
    if [[ -z "$JUDGE_LLM_API_KEY" ]]; then
        echo "Missing judge API key. Set JUDGE_LLM_API_KEY or $JUDGE_API_KEY_ENV." >&2
        missing=1
    fi
    return "$missing"
}

clean_for_answer_rerun() {
    local target_dir="$1"
    rm -f "$target_dir"/checkpoint_*.json
    rm -f "$target_dir"/responses_checkpoint_*.json
    rm -f "$target_dir"/answer_results.json
    rm -f "$target_dir"/answer_results.partial.json
    rm -f "$target_dir"/eval_results.json
    rm -f "$target_dir"/report.txt
}

copy_source_to_target() {
    local source_dir="$1"
    local target_dir="$2"

    if [[ -d "$target_dir" ]]; then
        if [[ "$FORCE" == "1" ]]; then
            echo "Force remove existing target: $target_dir"
            rm -rf "$target_dir"
        elif [[ "$RESUME_EXISTING" == "1" ]]; then
            echo "Resume existing target: $target_dir"
            return 0
        else
            echo "Target exists; skip. Set FORCE=1 to recreate or RESUME_EXISTING=1 to continue: $target_dir"
            return 2
        fi
    fi

    mkdir -p "$(dirname "$target_dir")"
    cp -a "$source_dir" "$target_dir"
    clean_for_answer_rerun "$target_dir"
}

run_one() {
    local dataset="$1"
    local system="$2"
    local mab_split="$3"
    local mab_source="$4"
    local source_run="$5"
    local selection="$6"
    local answer_model="$7"
    local judge_model="$8"

    local source_dir="$SOURCE_ROOT/$source_run"
    if [[ ! -d "$source_dir" ]]; then
        source_dir="$RESULTS_ROOT/$source_run"
    fi
    if [[ ! -d "$source_dir" ]]; then
        echo "Missing source directory for $source_run"
        return 1
    fi

    local answer_slug judge_slug target_dir run_label
    answer_slug="$(sanitize_component "$answer_model")"
    judge_slug="$(sanitize_component "$judge_model")"
    run_label="ans_${answer_slug}__judge_${judge_slug}__${SWEEP_TAG}"
    target_dir="$RESULTS_ROOT/${source_run}__${run_label}"

    echo ""
    echo "==> $selection"
    echo "    source: $source_run"
    echo "    target: $(basename "$target_dir")"
    echo "    answer: $answer_model"
    echo "    answer endpoint: $ANSWER_LLM_BASE_URL"
    echo "    judge:  $judge_model"
    echo "    judge endpoint:  $JUDGE_LLM_BASE_URL"
    if [[ "$dataset" != "memoryagentbench" ]]; then
        echo "    meme judge runs: $MEME_JUDGE_NUM_RUNS"
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi

    copy_source_to_target "$source_dir" "$target_dir"
    local copy_status=$?
    if [[ $copy_status -eq 2 ]]; then
        return 0
    elif [[ $copy_status -ne 0 ]]; then
        return $copy_status
    fi

    if [[ -s "$target_dir/eval_results.json" ]]; then
        echo "Target already has eval_results.json; skip: $target_dir"
        return 0
    fi

    local run_answer=1
    if [[ -s "$target_dir/answer_results.json" ]]; then
        echo "Target already has answer_results.json; run evaluate only: $target_dir"
        run_answer=0
    fi

    export PYTHONPATH="$EVERCORE_DIR:$EVERCORE_DIR/src:${PYTHONPATH:-}"

    local cmd=(python -m evaluation.cli --dataset "$dataset" --system "$system" --run-name "$run_label" --output-dir "$target_dir")

    if [[ "$dataset" == "memoryagentbench" ]]; then
        cmd+=(--mab-split "$mab_split" --mab-source "$mab_source")
    fi

    local answer_cmd=("${cmd[@]}" --stages answer)
    local evaluate_cmd=("${cmd[@]}" --stages evaluate)

    if [[ "$run_answer" == "1" ]]; then
        (
            cd "$EVERCORE_DIR" &&
            env \
                LLM_MODEL="$answer_model" \
                EMEM_LLM_MODEL="$answer_model" \
                EVALUATE_MODEL="$answer_model" \
                MEMORYAGENTBENCH_EVALUATE_MODEL="$answer_model" \
                LLM_API_KEY="$ANSWER_LLM_API_KEY" \
                LLM_BASE_URL="$ANSWER_LLM_BASE_URL" \
                "${answer_cmd[@]}"
        ) || return $?
    fi

    (
        cd "$EVERCORE_DIR" &&
        env \
            EVALUATE_MODEL="$judge_model" \
            MEMORYAGENTBENCH_EVALUATE_MODEL="$judge_model" \
            MEME_JUDGE_NUM_RUNS="$MEME_JUDGE_NUM_RUNS" \
            LLM_API_KEY="$JUDGE_LLM_API_KEY" \
            LLM_BASE_URL="$JUDGE_LLM_BASE_URL" \
            "${evaluate_cmd[@]}"
    )
}

main() {
    if [[ ! -f "$MANIFEST" ]]; then
        echo "Manifest not found: $MANIFEST" >&2
        return 1
    fi

    if [[ "$SKIP_CONDA_ACTIVATE" != "1" ]]; then
        if [[ -f "$CONDA_ACTIVATE" ]]; then
            # shellcheck disable=SC1090
            source "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
        else
            echo "Conda activate script not found: $CONDA_ACTIVATE" >&2
            echo "Set CONDA_ACTIVATE or SKIP_CONDA_ACTIVATE=1." >&2
            return 1
        fi
    fi

    load_env_file "$ENV_FILE"
    resolve_llm_endpoints
    if [[ "$DRY_RUN" != "1" ]]; then
        validate_llm_endpoints || return 1
    fi

    mkdir -p "$RESULTS_ROOT"
    initialize_sweep_tag

    local failures=0
    while IFS=$'\t' read -r enabled dataset system mab_split mab_source source_run selection _; do
        [[ -z "${enabled:-}" || "$enabled" == \#* ]] && continue
        [[ "$enabled" != "1" ]] && continue
        [[ "$mab_split" == "-" ]] && mab_split=""
        [[ "$mab_source" == "-" ]] && mab_source=""

        local judges=()
        if [[ "$dataset" == "memoryagentbench" ]]; then
            judges=("${MAB_JUDGE_MODEL_LIST[@]}")
        else
            judges=("${MEME_JUDGE_MODEL_LIST[@]}")
        fi

        local answer_model judge_model
        for answer_model in "${ANSWER_MODEL_LIST[@]}"; do
            for judge_model in "${judges[@]}"; do
                run_one "$dataset" "$system" "$mab_split" "$mab_source" "$source_run" "$selection" "$answer_model" "$judge_model"
                local status=$?
                if [[ $status -ne 0 ]]; then
                    echo "Failed ($status): $source_run answer=$answer_model judge=$judge_model" >&2
                    failures=$((failures + 1))
                    if [[ "$STOP_ON_ERROR" == "1" ]]; then
                        return "$status"
                    fi
                fi
            done
        done
    done < "$MANIFEST"

    if [[ $failures -gt 0 ]]; then
        echo "Completed with $failures failure(s)." >&2
        return 1
    fi
    clear_sweep_tag_state
    echo "Completed all scheduled reruns."
}

main "$@"
