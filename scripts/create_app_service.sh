#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SRC="$SCRIPT_DIR/app_scan.service"
SERVICE_DEST="$HOME/.config/systemd/user/app_scan.service"

mkdir -p "$(dirname "$SERVICE_DEST")"
cp "$SERVICE_SRC" "$SERVICE_DEST"
chmod 644 "$SERVICE_DEST"

systemctl --user daemon-reload
systemctl --user enable --now app_scan.service

echo "✅ App Scan user‑service installed & started."
