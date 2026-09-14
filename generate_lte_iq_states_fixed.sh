#!/usr/bin/env bash
set -Eeuo pipefail

# EMILY-X: one-command generation of four genuine srsRAN LTE downlink IQ states.
# Usage: ./generate_lte_iq_states_fixed.sh [terrestrial|d2c]
#
# Robust WSL version.  It intentionally does NOT use `setsid sudo ...` and does
# not depend on srsRAN C++ stdout being flushed into redirected log files.
# Readiness is detected from the actual network/ZMQ state instead:
#   EPC    -> srs_spgw_sgi exists
#   eNB    -> TCP/ZMQ port 2000 is listening
#   broker -> TCP/ZMQ port 2100 is listening
#   UE     -> tun_srsue has an IPv4 address inside the UE namespace
#   iperf  -> port 5201 is listening inside the UE namespace

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
BROKER="${BROKER:-$PROJECT_DIR/gnuradio_srsran_downlink_multicapture_v3.py}"

PROFILE="${1:-${PROFILE:-terrestrial}}"
case "$PROFILE" in
    terrestrial)
        DEFAULT_SAMPLE_RATE="11.52e6"
        DEFAULT_DL_EARFCN="6240"
        DEFAULT_N_PRB="50"
        DEFAULT_PREFIX="srsran"
        DEFAULT_LIGHT_RATE="5M"
        DEFAULT_MEDIUM_RATE="15M"
        PROFILE_LABEL="10 MHz / 50-PRB terrestrial LTE at 800 MHz"
        ;;
    d2c|leo)
        PROFILE="d2c"
        DEFAULT_SAMPLE_RATE="5.76e6"
        DEFAULT_DL_EARFCN="8665"
        DEFAULT_N_PRB="25"
        DEFAULT_PREFIX="srsran_d2c"
        DEFAULT_LIGHT_RATE="2M"
        DEFAULT_MEDIUM_RATE="7M"
        PROFILE_LABEL="5 MHz / 25-PRB D2C LTE at 1992.5 MHz"
        ;;
    *)
        echo "Usage: $0 [terrestrial|d2c]" >&2
        exit 2
        ;;
esac

UE_NS="${UE_NS:-ue1}"
SAMPLE_RATE="${SAMPLE_RATE:-$DEFAULT_SAMPLE_RATE}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-0.5}"
LIGHT_RATE="${LIGHT_RATE:-$DEFAULT_LIGHT_RATE}"
MEDIUM_RATE="${MEDIUM_RATE:-$DEFAULT_MEDIUM_RATE}"
TRAFFIC_SECONDS="${TRAFFIC_SECONDS:-7}"
TRAFFIC_SETTLE_SECONDS="${TRAFFIC_SETTLE_SECONDS:-2}"
IDLE_SETTLE_SECONDS="${IDLE_SETTLE_SECONDS:-2}"
DL_EARFCN="${DL_EARFCN:-$DEFAULT_DL_EARFCN}"
N_PRB="${N_PRB:-$DEFAULT_N_PRB}"
CAPTURE_PREFIX="${CAPTURE_PREFIX:-$DEFAULT_PREFIX}"

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$HOME/emilyx_lte_iq_$RUN_TAG}"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

EPC_PID=""
ENB_PID=""
BROKER_PID=""
UE_PID=""
IPERF_SERVER_PID=""
IPERF_CLIENT_PID=""
FIFO="$RUN_DIR/broker_commands.fifo"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

show_log_tail() {
    local file="$1"
    echo "---- tail of $file ----" >&2
    tail -n 100 "$file" 2>/dev/null >&2 || true
}

# Start a root-owned process without detaching sudo from the terminal first.
# A tiny root shell writes its own PID and then execs the requested command, so
# cleanup can target the actual srsRAN/iperf process rather than the sudo wrapper.
start_root_process() {
    local name="$1" logfile="$2" pidfile="$3"
    shift 3
    rm -f "$pidfile"

    sudo -n bash -c '
        pidfile="$1"
        user_home="$2"
        shift 2
        echo $$ > "$pidfile"
        exec env HOME="$user_home" "$@"
    ' _ "$pidfile" "$HOME" "$@" >"$logfile" 2>&1 &

    local sudo_wrapper_pid=$!
    local i
    for i in {1..30}; do
        if [[ -s "$pidfile" ]]; then
            cat "$pidfile"
            return 0
        fi
        if ! kill -0 "$sudo_wrapper_pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    show_log_tail "$logfile"
    die "$name failed before its PID could be established"
}

process_exists() {
    local pid="${1:-}"
    [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1
}

stop_pid() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    process_exists "$pid" || return 0

    kill -TERM "$pid" 2>/dev/null || sudo -n kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in {1..20}; do
        process_exists "$pid" || return 0
        sleep 0.1
    done
    kill -KILL "$pid" 2>/dev/null || sudo -n kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
    local rc=$?
    set +e
    if [[ -e "$FIFO" ]]; then
        printf 'q\n' >&3 2>/dev/null || true
    fi
    exec 3>&- 2>/dev/null || true

    stop_pid "$IPERF_CLIENT_PID"
    stop_pid "$IPERF_SERVER_PID"
    stop_pid "$UE_PID"
    stop_pid "$BROKER_PID"
    stop_pid "$ENB_PID"
    stop_pid "$EPC_PID"

    sudo -n ip netns del "$UE_NS" 2>/dev/null || sudo ip netns del "$UE_NS" 2>/dev/null || true

    if (( rc != 0 )); then
        echo >&2
        echo "Capture run failed. Logs are in: $LOG_DIR" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

for cmd in srsepc srsenb srsue iperf3 python3 ip ss ping awk grep sed cp ps; do
    require_cmd "$cmd"
done
[[ -f "$BROKER" ]] || die "Broker not found: $BROKER"

log "Authenticating sudo once"
sudo -v

log "Preparing clean UE network namespace '$UE_NS'"
sudo ip netns del "$UE_NS" 2>/dev/null || true
sudo ip netns add "$UE_NS"
sudo rm -f /tmp/epc.log 2>/dev/null || true

# Readiness predicates.  These interrogate actual OS/network state and therefore
# do not care whether srsRAN has buffered its redirected console output.
epc_ready() {
    ip -4 addr show dev srs_spgw_sgi 2>/dev/null | grep -q 'inet 172\.16\.0\.1/'
}

enb_ready() {
    ss -ltnH 2>/dev/null | awk '$4 ~ /:2000$/ {found=1} END {exit !found}'
}

broker_ready() {
    ss -ltnH 2>/dev/null | awk '$4 ~ /:2100$/ {found=1} END {exit !found}'
}

ue_ready() {
    sudo -n ip netns exec "$UE_NS" ip -4 -o addr show dev tun_srsue 2>/dev/null | grep -q ' inet '
}

iperf_server_ready() {
    sudo -n ip netns exec "$UE_NS" ss -ltnH 2>/dev/null | awk '$4 ~ /:5201$/ {found=1} END {exit !found}'
}

wait_ready() {
    local description="$1" timeout_s="$2" pid="$3" logfile="$4" predicate="$5"
    local loops=$((timeout_s * 10))
    local i
    for ((i=0; i<loops; i++)); do
        if "$predicate"; then
            return 0
        fi
        if [[ -n "$pid" ]] && ! process_exists "$pid"; then
            echo "$description process exited before becoming ready." >&2
            show_log_tail "$logfile"
            return 1
        fi
        sleep 0.1
    done
    echo "Timed out waiting for: $description" >&2
    show_log_tail "$logfile"
    return 1
}

wait_for_log() {
    local file="$1" pattern="$2" timeout_s="$3"
    local t=0
    while (( t < timeout_s * 10 )); do
        if [[ -f "$file" ]] && grep -Fq "$pattern" "$file"; then
            return 0
        fi
        sleep 0.1
        ((t+=1))
    done
    echo "Timed out waiting for: $pattern" >&2
    show_log_tail "$file"
    return 1
}

log "Starting srsEPC"
EPC_PID="$(start_root_process \
    srsEPC \
    "$LOG_DIR/epc.log" \
    "$PID_DIR/epc.pid" \
    srsepc)"
wait_ready "srsEPC / srs_spgw_sgi" 20 "$EPC_PID" "$LOG_DIR/epc.log" epc_ready \
    || die "srsEPC did not initialize"
log "srsEPC ready (srs_spgw_sgi = 172.16.0.1)"

log "Starting $PROFILE_LABEL (EARFCN $DL_EARFCN)"
srsenb \
    --enb.n_prb="$N_PRB" \
    --enb.tm=1 \
    --enb.nof_ports=1 \
    --rf.dl_earfcn="$DL_EARFCN" \
    --rf.device_name=zmq \
    --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2000,rx_port=tcp://localhost:2001,id=enb,base_srate=$SAMPLE_RATE" \
    >"$LOG_DIR/enb.log" 2>&1 &
ENB_PID=$!
wait_ready "srsENB ZMQ TX port 2000" 20 "$ENB_PID" "$LOG_DIR/enb.log" enb_ready \
    || die "srsENB did not start"
log "srsENB ready on ZMQ port 2000"

log "Starting transparent GNU Radio downlink broker"
mkfifo "$FIFO"
# Keep one read/write descriptor open for the lifetime of the broker so input()
# never sees EOF between automated capture commands.
exec 3<>"$FIFO"
python3 -u "$BROKER" \
    --enb-tx tcp://127.0.0.1:2000 \
    --ue-rx 'tcp://*:2100' \
    --sample-rate "$SAMPLE_RATE" \
    --capture-seconds "$CAPTURE_SECONDS" \
    --output-dir "$RUN_DIR" \
    --prefix "$CAPTURE_PREFIX" \
    <"$FIFO" >"$LOG_DIR/broker.log" 2>&1 &
BROKER_PID=$!
wait_ready "GNU Radio broker ZMQ UE port 2100" 20 "$BROKER_PID" "$LOG_DIR/broker.log" broker_ready \
    || die "GNU Radio broker did not start"
log "GNU Radio broker ready on ZMQ port 2100"

log "Starting srsUE through the broker"
UE_PID="$(start_root_process \
    srsUE \
    "$LOG_DIR/ue.log" \
    "$PID_DIR/ue.pid" \
    srsue \
    --rf.device_name=zmq \
    --rf.device_args="tx_port=tcp://*:2001,rx_port=tcp://localhost:2100,id=ue,base_srate=$SAMPLE_RATE" \
    --rat.eutra.dl_earfcn="$DL_EARFCN" \
    --gw.netns="$UE_NS")"
wait_ready "UE attach / tun_srsue IPv4 address" 35 "$UE_PID" "$LOG_DIR/ue.log" ue_ready \
    || die "UE did not attach"

UE_IP="$(sudo -n ip netns exec "$UE_NS" ip -4 -o addr show dev tun_srsue | awk '{split($4,a,"/"); print a[1]; exit}')"
[[ -n "$UE_IP" ]] || die "Could not determine UE IP address from tun_srsue"
log "UE attached at $UE_IP"

log "Verifying LTE user plane"
for _ in {1..10}; do
    if ping -c 1 -W 2 "$UE_IP" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
ping -c 1 -W 3 "$UE_IP" >/dev/null 2>&1 || die "Could not ping UE at $UE_IP"
log "LTE user plane verified by ping"

log "Starting persistent iperf3 server inside namespace $UE_NS"
IPERF_SERVER_PID="$(start_root_process \
    iperf3-server \
    "$LOG_DIR/iperf_server.log" \
    "$PID_DIR/iperf_server.pid" \
    ip netns exec "$UE_NS" iperf3 -s)"
wait_ready "iperf3 server on UE port 5201" 10 "$IPERF_SERVER_PID" "$LOG_DIR/iperf_server.log" iperf_server_ready \
    || die "iperf3 server did not start"

capture_state() {
    local state="$1"
    log "Capturing LTE state: $state"
    printf '%s\n' "$state" >&3
    wait_for_log "$LOG_DIR/broker.log" "Complete: ${CAPTURE_PREFIX}_${state}.cf32" 20 \
        || die "Capture '$state' did not complete"
    [[ -s "$RUN_DIR/${CAPTURE_PREFIX}_${state}.cf32" ]] || die "Capture file is empty: $state"
}

run_udp_and_capture() {
    local state="$1" rate="$2"
    log "Starting $state downlink traffic at $rate UDP"
    iperf3 -c "$UE_IP" -u -b "$rate" -t "$TRAFFIC_SECONDS" \
        >"$LOG_DIR/iperf_${state}.log" 2>&1 &
    IPERF_CLIENT_PID=$!
    sleep "$TRAFFIC_SETTLE_SECONDS"
    kill -0 "$IPERF_CLIENT_PID" 2>/dev/null || {
        cat "$LOG_DIR/iperf_${state}.log" >&2
        die "iperf3 $state traffic stopped before capture"
    }
    capture_state "$state"
    if ! wait "$IPERF_CLIENT_PID"; then
        cat "$LOG_DIR/iperf_${state}.log" >&2
        die "iperf3 $state traffic failed"
    fi
    IPERF_CLIENT_PID=""
    sleep 1
}

# 1) Idle: no user traffic.
log "Allowing cell to settle to idle"
sleep "$IDLE_SETTLE_SECONDS"
capture_state idle
sleep 1

# 2) Controlled light and medium offered loads.
run_udp_and_capture light "$LIGHT_RATE"
run_udp_and_capture medium "$MEDIUM_RATE"

# 3) Loaded: same style as the previously validated ~31 Mbit/s test.
log "Starting saturated loaded downlink with 4 parallel TCP streams"
iperf3 -c "$UE_IP" -t "$TRAFFIC_SECONDS" -P 4 \
    >"$LOG_DIR/iperf_loaded.log" 2>&1 &
IPERF_CLIENT_PID=$!
sleep "$TRAFFIC_SETTLE_SECONDS"
kill -0 "$IPERF_CLIENT_PID" 2>/dev/null || {
    cat "$LOG_DIR/iperf_loaded.log" >&2
    die "loaded iperf3 traffic stopped before capture"
}
capture_state loaded
if ! wait "$IPERF_CLIENT_PID"; then
    cat "$LOG_DIR/iperf_loaded.log" >&2
    die "loaded iperf3 traffic failed"
fi
IPERF_CLIENT_PID=""

log "Adding automated traffic-profile metadata"
python3 - "$RUN_DIR" "$LIGHT_RATE" "$MEDIUM_RATE" "$CAPTURE_PREFIX" "$PROFILE" "$N_PRB" "$DL_EARFCN" "$SAMPLE_RATE" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
light_rate = sys.argv[2]
medium_rate = sys.argv[3]
prefix = sys.argv[4]
profile_name = sys.argv[5]
n_prb = int(sys.argv[6])
dl_earfcn = int(sys.argv[7])
sample_rate = float(sys.argv[8])
profiles = {
    "idle": {
        "traffic_generator": "none",
        "offered_downlink_load": "idle cell; no user iperf traffic",
    },
    "light": {
        "traffic_generator": "iperf3 UDP",
        "offered_downlink_load": light_rate,
    },
    "medium": {
        "traffic_generator": "iperf3 UDP",
        "offered_downlink_load": medium_rate,
    },
    "loaded": {
        "traffic_generator": "iperf3 TCP",
        "offered_downlink_load": "4 parallel TCP streams; saturation test",
    },
}
for state, profile in profiles.items():
    p = run_dir / f"{prefix}_{state}.cf32.json"
    d = json.loads(p.read_text())
    d["automated_capture"] = True
    d["traffic_profile"] = profile
    d["radio_profile"] = profile_name
    d["lte_n_prb"] = n_prb
    d["lte_dl_earfcn"] = dl_earfcn
    d["sample_rate_hz"] = sample_rate
    p.write_text(json.dumps(d, indent=2) + "\n")
PY

log "Validating file sizes and measuring relative complex RMS"
python3 - "$RUN_DIR" "$SAMPLE_RATE" "$CAPTURE_SECONDS" "$CAPTURE_PREFIX" <<'PY'
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
fs = float(sys.argv[2])
dur = float(sys.argv[3])
prefix = sys.argv[4]
expected_samples = round(fs * dur)
expected_bytes = expected_samples * np.dtype(np.complex64).itemsize
states = ["idle", "light", "medium", "loaded"]
rms = {}
for state in states:
    p = root / f"{prefix}_{state}.cf32"
    size = p.stat().st_size
    if size != expected_bytes:
        raise SystemExit(f"{p.name}: expected {expected_bytes} bytes, got {size}")
    x = np.memmap(p, mode="r", dtype=np.complex64)
    rms[state] = float(np.sqrt(np.mean(np.abs(x.astype(np.complex128))**2)))
ref = rms["loaded"]
print("\nCapture summary")
print("---------------")
for state in states:
    rel = 20*np.log10(rms[state]/ref) if rms[state] > 0 and ref > 0 else float("nan")
    print(f"{state:7s}  {expected_samples:,} samples  {expected_bytes/1e6:6.1f} MB  RMS={rms[state]:.6g}  rel_loaded={rel:+.2f} dB")
PY

log "Copying captures and sidecars to Windows/project directory"
for state in idle light medium loaded; do
    cp -f "$RUN_DIR/${CAPTURE_PREFIX}_${state}.cf32" "$PROJECT_DIR/"
    cp -f "$RUN_DIR/${CAPTURE_PREFIX}_${state}.cf32.json" "$PROJECT_DIR/"
done

log "Done"
echo "Captures copied to: $PROJECT_DIR"
echo "Run logs retained in: $LOG_DIR"
echo
printf '  %s\n' \
    "$PROJECT_DIR/${CAPTURE_PREFIX}_idle.cf32" \
    "$PROJECT_DIR/${CAPTURE_PREFIX}_light.cf32" \
    "$PROJECT_DIR/${CAPTURE_PREFIX}_medium.cf32" \
    "$PROJECT_DIR/${CAPTURE_PREFIX}_loaded.cf32"

# Normal exit triggers cleanup, which shuts down the LTE stack and namespace.
