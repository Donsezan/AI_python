# Oracle Cloud VPS Setup — Ubuntu 24.04 Minimal

## Initial Setup

### 1. Connect to your VM

```bash
ssh -i "D:\Private dock\Scans\Keys\Oracle\PythonAI\ssh-key-2026-05-16.key" ubuntu@YOUR_PUBLIC_IP
```

### 2. Update system & install dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y
```

### 3. Clone your repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 4. Create virtualenv & install packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Create `.env` file with credentials

```bash
nano .env
```

Paste all your credentials:

```env
BOT_TOKEN=...
CHAT_ID=...
NEWS_URL=...
GEMINI_API_KEY=...
SUPABASE_URL=...
SUPABASE_KEY=...
# Optional:
# GEMINI_MODEL=gemini-2.5-flash-lite
# GEMINI_MIN_CALL_INTERVAL_SEC=6.5
# LOG_LEVEL=INFO
```

Save with `Ctrl+X` → `Y` → `Enter`.

### 6. Test the bot runs

```bash
python3 main.py --dry-run
```

If no errors, stop it with `Ctrl+C` and proceed.

---

## Running as a System Service

### 7. Create systemd service

```bash
sudo nano /etc/systemd/system/newsbot.service
```

Paste the following (replace `YOUR_REPO` with your actual folder name):

```ini
[Unit]
Description=News Bot
After=network.target
# Stop restart-looping if the bot crashes 5 times within 10 minutes
# (e.g. a missing .env variable raises at startup)
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/YOUR_REPO
ExecStart=/home/ubuntu/YOUR_REPO/.venv/bin/python main.py
Restart=on-failure
RestartSec=30
# Keep a runaway process from eating the Always Free instance's RAM
MemoryMax=512M

[Install]
WantedBy=multi-user.target
```

Save with `Ctrl+X` → `Y` → `Enter`.

### 8. Enable & start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable newsbot
sudo systemctl start newsbot
sudo systemctl status newsbot
```

You should see `Active: active (running)`.

### 9. Monitor live logs

```bash
journalctl -u newsbot -f
```

The bot now runs 24/7, auto-starts on reboot, and auto-restarts on crash.

---

## Stopping & Updating the Bot

### Stop the service

```bash
sudo systemctl stop newsbot
```

### Make your code changes

**Option A — Pull latest from Git:**

```bash
cd ~/YOUR_REPO
git pull
```

**Option B — Edit a file directly:**

```bash
nano main.py
```

**Option C — Reinstall dependencies** (if `requirements.txt` changed):

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Restart the service

```bash
sudo systemctl start newsbot
```

### Verify it's running

```bash
sudo systemctl status newsbot
```
