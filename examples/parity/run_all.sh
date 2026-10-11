#!/bin/bash
# Run every parity example: bash run_all.sh OUT_DIR [GPU_ID] [EXAMPLE...]
# Each example writes OUT_DIR/<name>/ (the -o folder) and OUT_DIR/<name>/stdout.txt.
# VESPER is the command that runs the CLI (default: vesper); set it to compare another engine.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$1
gpu=${2:-0}
shift $(($# < 2 ? $# : 2))
read -r -a vesper <<<"${VESPER:-vesper}"
in=$out/inputs

[ -f "$in/ca.pdb" ] || python "$here/make_inputs.py" "$in" || exit 1

# the map, the structure simulated at 5 A, and a quick search
base=(-a "$in/map.mrc" -t 0.0289 -b "$in/model.pdb" -T 0.01 -res 5 -s 2 -g 2 -A 30 -N 5 -c 8)
gpu_base=("${base[@]}" -gpu "$gpu")

declare -A args
args[search_V]="-M V -pdbin $in/model.pdb"
args[search_C]="-M C -pdbin $in/model.pdb"
args[search_P]="-M P -pdbin $in/model.pdb"
args[search_O]="-M O -pdbin $in/model.pdb"
args[search_L]="-M L -pdbin $in/model.pdb"
args[refine_R1]="-M V -R 1 -pdbin $in/model.pdb"
args[nodup]="-M V -N 10 -nodup -pdbin $in/model.pdb"
args[pdbin_cif]="-M V -pdbin $in/model.cif"
args[batch_3]="-M V -batch 3 -pdbin $in/model.pdb"
args[multi_map]="-a $in/map.mrc,$in/half.mrc -labels full,half -M C -pdbin $in/model.pdb"
args[half_alone]="-a $in/half.mrc -M C -pdbin $in/model.pdb"
args[ldp]="-M C -ldp $in/ldp.pdb -ca $in/ca.pdb -pdbin $in/model.pdb"
args[direct_fit]="-M C --direct_fit -pdbin $in/model.pdb"
args[local_fit]="-M C -A 1 -al 20 -N 1 -ldp $in/ldp.pdb -ca $in/ca.pdb -pdbin $in/model.pdb"
args[eval_half]="-E -b $in/half.mrc"
args[eval_moved]="-E -b $in/moved.mrc"
args[vectors]="-v $out/vectors/vectors"
order=(search_V search_C search_P search_O search_L refine_R1 nodup pdbin_cif batch_3 multi_map
  half_alone ldp direct_fit local_fit eval_half eval_moved vectors diffmodeler cpu)

run() {  # run NAME ARGS...: one example, its exit status in OUT_DIR/<name>/status.txt
  local name=$1
  shift
  rm -rf "${out:?}/$name"
  mkdir -p "$out/$name"
  local start=$SECONDS
  "${vesper[@]}" orig "$@" -o "$out/$name" >"$out/$name/stdout.txt" 2>&1
  local status=$?
  echo "exit=$status seconds=$((SECONDS - start))" >"$out/$name/status.txt"
  echo "$name: exit $status, $((SECONDS - start)) s"
}

for name in "${@:-${order[@]}}"; do
  case $name in
  diffmodeler)
    # DiffModeler's global fit of one chain against two maps (config/diffmodeler.json)
    run diffmodeler -a "$in/map.mrc,$in/half.mrc" -labels full,half -t 0.05 -b "$in/model.pdb" \
      -T 0.01 -g 2 -s 2 -A 10 -N 100 -M C -gpu "$gpu" -pdbin "$in/model.pdb" \
      -ldp "$in/ldp.pdb" -ca "$in/ca.pdb" -c 8 -res 1.0
    ;;
  cpu)
    # CPU search (no -gpu), coarser grid to stay quick
    run cpu "${base[@]}" -s 4 -M V -pdbin "$in/model.pdb"
    ;;
  *)
    [ -n "${args[$name]+x}" ] || { echo "unknown example $name"; exit 2; }
    read -r -a extra <<<"${args[$name]}"
    run "$name" "${gpu_base[@]}" "${extra[@]}"
    ;;
  esac
done
