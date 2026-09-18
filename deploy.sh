#!/bin/bash
set -ex

# The service file goes here:
#   ~/.config/systemd/user/ai.service
# Set it up with:
#   systemctl --user daemon-reload
#   systemctl --user enable --now ai
#   systemctl --user start ai
#   loginctl enable-linger $USER

cd "$(dirname "$0")"
go build -o ai ./cmd/serve
cat ./ai.service | sed "s|###|$(pwd)/ai|" > ~/.config/systemd/user/ai.service
systemctl --user daemon-reload
systemctl --user restart ai.service
journalctl --user -n 100 -f -u ai
