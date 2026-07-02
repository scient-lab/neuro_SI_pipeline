#!/usr/bin/env bash
# s3_run_size.sh — human-readable size of an S3 prefix you pass in directly.
#
# One `aws s3 ls --recursive` call; totals summed locally, so --breakdown
# (per first-level folder under the prefix) costs no extra API calls.
#
# Usage:
#   ./scripts/s3_size.sh s3://enlibra/dss/runs/<RUN_ID>/
#   ./scripts/s3_size.sh s3://enlibra/dss/runs/<RUN_ID>/ -b        # + folder breakdown
#   ./scripts/s3_size.sh s3://enlibra/dss/runs/<RUN_ID>/outputs/ -b
#   ./scripts/s3_size.sh s3://.../ --profile myprofile
#
# Exit: 0 ok, 1 usage/aws error.
set -euo pipefail

URI=""
BREAKDOWN=0
PROFILE_FLAG=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--breakdown) BREAKDOWN=1; shift ;;
        --profile)      PROFILE_FLAG=(--profile "$2"); shift 2 ;;
        -h|--help)      sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \?//'; exit 0 ;;
        -*)             echo "unknown flag: $1" >&2; exit 1 ;;
        *)              URI="$1"; shift ;;
    esac
done

[[ -z "$URI" ]] && { echo "usage: $0 s3://bucket/prefix/ [-b] [--profile P]" >&2; exit 1; }
[[ "$URI" == s3://* ]] || { echo "path must start with s3:// — got '$URI'" >&2; exit 1; }
URI="${URI%/}/"                       # normalize one trailing slash

# key portion after s3://bucket/ (empty if the URI is a bare bucket)
_tmp="${URI#s3://}"; KEYPREFIX="${_tmp#*/}"; [[ "$KEYPREFIX" == "$_tmp" ]] && KEYPREFIX=""

listing=$(aws "${PROFILE_FLAG[@]}" s3 ls "$URI" --recursive 2>/dev/null || true)
if [[ -z "$listing" ]]; then
    echo "Path    : $URI"; echo "(empty — no objects under this prefix)"; exit 0
fi

# $3 = size in bytes; key = everything after "DATE TIME SIZE ". Group by the
# first path component under KEYPREFIX (substr compare, not regex — paths have
# dots/dashes).
agg=$(printf '%s\n' "$listing" | awk -v kp="$KEYPREFIX" '
    { sz=$3; key=$0; sub(/^[^ ]+ +[^ ]+ +[0-9]+ +/, "", key);
      if (kp != "" && substr(key,1,length(kp)) == kp) key=substr(key, length(kp)+1);
      p=index(key, "/"); g=(p ? substr(key,1,p-1) : "(files)");
      tot[g]+=sz; sum+=sz; cnt++ }
    # %.0f, NOT %d: byte totals exceed 2^31 and mawk %d saturates at 32-bit.
    END { printf "SUM\t%.0f\t%d\n", sum, cnt;
          for (g in tot) printf "GRP\t%.0f\t%s\n", tot[g], g }')

_h() { numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1}B"; }

sum=$(awk -F'\t' '$1=="SUM"{print $2}' <<<"$agg")
cnt=$(awk -F'\t' '$1=="SUM"{print $3}' <<<"$agg")
echo "Path    : $URI"
echo "Objects : $cnt"
echo "Total   : $(_h "$sum")"

if [[ "$BREAKDOWN" -eq 1 ]]; then
    echo
    awk -F'\t' '$1=="GRP"{print $2"\t"$3}' <<<"$agg" | sort -rn | \
    while IFS=$'\t' read -r bytes grp; do printf '  %10s  %s\n' "$(_h "$bytes")" "$grp"; done
fi
