#!/usr/bin/env bash
# Run inside the pilot directory on the instance after authenticating to both datasets.
set -euo pipefail
umask 077
cd "$(dirname "$0")"
if [[ -f /etc/network_turbo ]]; then
    source /etc/network_turbo >/dev/null
fi
export HF_ENDPOINT=https://huggingface.co
mkdir -p logs
dest="${1:-$PWD/metadata}"
py="$PWD/.hf-venv/bin/python"
"$py" fetch_manifest.py --download --destination "$dest" >logs/metadata_download.json
report_dir="$dest/ct_rate/dataset/radiology_text_reports"
summary=$(mktemp "$PWD/logs/pair_audit.XXXXXX")
if "$py" audit_pairs.py \
    --rex "$dest/rex/dataset.json" \
    --extra-holdout "$dest/rex/MICCAI_challenge_dataset.json" \
    --reports "$report_dir/train_reports.csv" "$report_dir/validation_reports.csv" >"$summary"
then
    mv "$summary" logs/pair_audit.json
    cat logs/pair_audit.json
else
    mv "$summary" logs/pair_audit_incomplete.json
    echo 'Real metadata audit incomplete. See logs/pair_audit_incomplete.json and the error above.' >&2
    exit 1
fi
