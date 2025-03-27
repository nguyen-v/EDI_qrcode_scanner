#!/bin/bash
# This script stops, disables, and deletes the app_scan.service user‑level systemd service.

SERVICE_FILE="$HOME/.config/systemd/user/app_scan.service"

# Check if the service file exists
if [[ -f "$SERVICE_FILE" ]]; then
    # Stop & disable via user‑service
    systemctl --user stop app_scan.service
    systemctl --user disable app_scan.service

    # Remove the service file
    rm -f "$SERVICE_FILE"

    # Reload user daemon
    systemctl --user daemon-reload

    echo "✅ App Scan user service removed successfully."
    exit 0
else
    echo "❌ app_scan.service not found at $SERVICE_FILE"
    exit 1
fi
