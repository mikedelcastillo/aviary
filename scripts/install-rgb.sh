#!/usr/bin/env bash
#
# scripts/install-rgb.sh — provision OpenRGB so aviary can drive the motherboard /
# header / RAM RGB as a live status display (see server/lib/rgb/).
#
# What it does (idempotent, safe to re-run):
#   1. Downloads + extracts the OpenRGB AppImage to ~/.local/share/aviary/openrgb
#      (no FUSE needed — we extract the squashfs once and run the binary directly).
#   2. Installs OpenRGB's udev rules so your user can reach the HID + i2c/SMBus
#      controllers WITHOUT running as root  (the only step that needs sudo).
#   3. Loads the i2c-dev kernel module (and persists it) so RGB RAM / addressable
#      headers on the SMBus can be detected.
#   4. Installs a *user* systemd service that keeps the OpenRGB SDK server
#      (127.0.0.1:6742) running at login — aviary connects to it as a client.
#   5. Adds `openrgb-python` to the uv project and verifies device enumeration.
#
# Usage (from the repo root):
#   ./scripts/install-rgb.sh                # full setup
#   ./scripts/install-rgb.sh --no-sudo      # skip the steps that need root (udev/i2c/linger)
#   ./scripts/install-rgb.sh --no-service   # set up OpenRGB but don't install the systemd unit
#   ./scripts/install-rgb.sh --uninstall    # remove the user service (leaves OpenRGB + udev)
#
set -euo pipefail

# ---- config -----------------------------------------------------------------
OPENRGB_VERSION="1.0rc3"
OPENRGB_HASH="6fbcf62"
OPENRGB_URL="https://codeberg.org/OpenRGB/OpenRGB/releases/download/release_candidate_${OPENRGB_VERSION}/OpenRGB_${OPENRGB_VERSION}_x86_64_${OPENRGB_HASH}.AppImage"
SERVER_HOST="127.0.0.1"
SERVER_PORT="6742"

DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/aviary/openrgb"
BIN_DIR="${HOME}/.local/bin"
APPIMAGE="${DATA_DIR}/OpenRGB.AppImage"
EXTRACT_DIR="${DATA_DIR}/squashfs-root"
OPENRGB_BIN="${EXTRACT_DIR}/usr/bin/OpenRGB"
SERVICE_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${SERVICE_DIR}/aviary-openrgb.service"

USE_SUDO=1
USE_SERVICE=1
UNINSTALL=0
for arg in "$@"; do
  case "$arg" in
    --no-sudo) USE_SUDO=0 ;;
    --no-service) USE_SERVICE=0 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

c_ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
c_info() { printf '\033[36m•\033[0m %s\n' "$*"; }
c_warn() { printf '\033[33m!\033[0m %s\n' "$*"; }
c_err()  { printf '\033[31m✗\033[0m %s\n' "$*" >&2; }

# The script lives in scripts/, so the repo root is one level up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- uninstall path ---------------------------------------------------------
if [[ "$UNINSTALL" == 1 ]]; then
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user disable --now aviary-openrgb.service 2>/dev/null || true
  fi
  rm -f "$SERVICE_FILE"
  systemctl --user daemon-reload 2>/dev/null || true
  c_ok "Removed user service. (OpenRGB binary + udev rules left in place.)"
  exit 0
fi

echo "=============================================="
echo " aviary RGB / OpenRGB setup"
echo "=============================================="

# ---- 0. prerequisites -------------------------------------------------------
for tool in curl uname; do
  command -v "$tool" >/dev/null 2>&1 || { c_err "missing required tool: $tool"; exit 1; }
done
if [[ "$(uname -m)" != "x86_64" ]]; then
  c_warn "This script pins the x86_64 OpenRGB build; your arch is $(uname -m). Edit OPENRGB_URL if needed."
fi

maybe_sudo() {
  # Run a privileged command, honoring --no-sudo.
  if [[ "$USE_SUDO" == 0 ]]; then c_warn "skipping (--no-sudo): $*"; return 0; fi
  if [[ "$(id -u)" == 0 ]]; then "$@"; else sudo "$@"; fi
}

# ---- 1. download + extract OpenRGB -----------------------------------------
if command -v openrgb >/dev/null 2>&1; then
  OPENRGB_BIN="$(command -v openrgb)"
  c_ok "Found system OpenRGB at ${OPENRGB_BIN} — using it."
else
  mkdir -p "$DATA_DIR" "$BIN_DIR"
  if [[ ! -x "$OPENRGB_BIN" ]]; then
    c_info "Downloading OpenRGB ${OPENRGB_VERSION} …"
    curl -fL --retry 3 -o "$APPIMAGE" "$OPENRGB_URL"
    chmod +x "$APPIMAGE"
    c_info "Extracting (no FUSE needed) …"
    ( cd "$DATA_DIR" && "$APPIMAGE" --appimage-extract >/dev/null )
    c_ok "OpenRGB extracted to ${EXTRACT_DIR}"
  else
    c_ok "OpenRGB already extracted at ${EXTRACT_DIR}"
  fi
  # convenience symlink
  ln -sf "$OPENRGB_BIN" "${BIN_DIR}/openrgb"
  c_info "Symlinked ${BIN_DIR}/openrgb -> OpenRGB"
  case ":$PATH:" in *":$BIN_DIR:"*) ;; *) c_warn "Add ${BIN_DIR} to your PATH to use 'openrgb' directly." ;; esac
fi

# ---- 2. udev rules (root) ---------------------------------------------------
RULES_SRC="${EXTRACT_DIR}/usr/lib/udev/rules.d/60-openrgb.rules"
if [[ -f /etc/udev/rules.d/60-openrgb.rules ]]; then
  c_ok "udev rules already installed (/etc/udev/rules.d/60-openrgb.rules)."
elif [[ -f "$RULES_SRC" ]]; then
  c_info "Installing OpenRGB udev rules (lets your user reach the controllers without root)…"
  maybe_sudo cp "$RULES_SRC" /etc/udev/rules.d/60-openrgb.rules
  maybe_sudo udevadm control --reload-rules || true
  maybe_sudo udevadm trigger || true
  c_ok "udev rules installed. (You may need to replug USB RGB / reboot once.)"
else
  c_warn "Could not find bundled udev rules; controllers may require sudo until rules are added."
fi

# ---- 3. i2c-dev (root) for RAM / addressable SMBus controllers --------------
if lsmod 2>/dev/null | grep -q '^i2c_dev'; then
  c_ok "i2c-dev already loaded."
else
  c_info "Loading i2c-dev (needed to probe RGB RAM / SMBus controllers)…"
  maybe_sudo modprobe i2c-dev || c_warn "could not modprobe i2c-dev"
fi
if [[ "$USE_SUDO" == 1 && ! -f /etc/modules-load.d/i2c-dev.conf ]]; then
  echo 'i2c-dev' | maybe_sudo tee /etc/modules-load.d/i2c-dev.conf >/dev/null || true
  c_ok "i2c-dev set to load at boot."
fi

# ---- 4. systemd user service for the SDK server -----------------------------
if [[ "$USE_SERVICE" == 1 ]] && command -v systemctl >/dev/null 2>&1; then
  mkdir -p "$SERVICE_DIR"
  cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=OpenRGB SDK server for aviary (LED status display)
After=graphical-session.target

[Service]
Type=simple
ExecStart=${OPENRGB_BIN} --server --server-host ${SERVER_HOST} --server-port ${SERVER_PORT}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
  c_ok "Wrote ${SERVICE_FILE}"
  # Guard like the rest of the script: over SSH / headless the user systemd
  # instance or session bus may be unreachable; warn instead of aborting under set -e.
  systemctl --user daemon-reload 2>/dev/null \
    || c_warn "could not daemon-reload user systemd (user instance not reachable?)."
  systemctl --user enable --now aviary-openrgb.service 2>/dev/null \
    || c_warn "could not enable user service — start it in a user session or enable lingering."
  # Keep it running even when no desktop session is active (needs root once).
  if ! loginctl show-user "$USER" 2>/dev/null | grep -q 'Linger=yes'; then
    c_info "Enabling lingering so the LED server runs without an active login…"
    maybe_sudo loginctl enable-linger "$USER" || c_warn "could not enable linger (server still runs while logged in)."
  fi
  sleep 2
  if ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT} "; then
    c_ok "OpenRGB SDK server is listening on ${SERVER_HOST}:${SERVER_PORT}."
  else
    c_warn "Server not listening yet — check: journalctl --user -u aviary-openrgb -n 30"
  fi
else
  c_info "Skipping systemd service. Start the server manually with:"
  echo "    ${OPENRGB_BIN} --server --server-host ${SERVER_HOST} --server-port ${SERVER_PORT}"
fi

# ---- 5. python client + verify ---------------------------------------------
if command -v uv >/dev/null 2>&1; then
  c_info "Ensuring openrgb-python is in the uv project…"
  ( cd "$REPO_ROOT" && uv add 'openrgb-python>=0.3.2' >/dev/null 2>&1 ) \
    && c_ok "openrgb-python added." \
    || c_warn "could not 'uv add openrgb-python' automatically — add it to pyproject deps."
fi

echo
c_info "Enumerating detected RGB devices:"
"$OPENRGB_BIN" --list-devices 2>/dev/null | grep -E '^[0-9]+:|Zones:|LEDs:' | sed 's/^/    /' || true

echo
echo "=============================================="
c_ok "Done. Next:"
echo "    uv sync                 # install deps incl. openrgb-python"
echo "    uv run rgb-calibrate    # map each physical LED to a bird in the roster"
echo "    uv run rgb-demo         # preview the scenes (boot/discovery/birds/night)"
echo "=============================================="
