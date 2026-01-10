#!/bin/bash
# VPS Setup Script for Polymarket Analytics
# Run as root on a fresh Ubuntu/Debian server

set -e

echo "=== Polymarket Analytics VPS Setup ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./setup-vps.sh)"
    exit 1
fi

# Create polymarket user
echo "[1/8] Creating polymarket user..."
if ! id "polymarket" &>/dev/null; then
    useradd -m -s /bin/bash polymarket
    echo "Created user 'polymarket'"
else
    echo "User 'polymarket' already exists"
fi

# Install system dependencies
echo "[2/8] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    sqlite3 \
    curl \
    htop \
    logrotate

# Create directories
echo "[3/8] Creating directories..."
mkdir -p /opt/polymarket-analytics
mkdir -p /var/log/polymarket
chown polymarket:polymarket /opt/polymarket-analytics
chown polymarket:polymarket /var/log/polymarket

# Clone or update repository
echo "[4/8] Setting up repository..."
if [ -d "/opt/polymarket-analytics/.git" ]; then
    echo "Repository exists, pulling latest..."
    sudo -u polymarket git -C /opt/polymarket-analytics pull
else
    echo "Cloning repository..."
    # NOTE: Update this URL to your actual repo
    echo "Please clone your repository to /opt/polymarket-analytics"
    echo "Example: git clone https://github.com/YOUR_USER/polymarket-analytics.git /opt/polymarket-analytics"
fi

# Create virtual environment
echo "[5/8] Setting up Python virtual environment..."
sudo -u polymarket python3.10 -m venv /opt/polymarket-analytics/venv
sudo -u polymarket /opt/polymarket-analytics/venv/bin/pip install --upgrade pip

# Install Python dependencies
echo "[6/8] Installing Python dependencies..."
if [ -f "/opt/polymarket-analytics/requirements.txt" ]; then
    sudo -u polymarket /opt/polymarket-analytics/venv/bin/pip install -r /opt/polymarket-analytics/requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

# Install systemd services
echo "[7/8] Installing systemd services..."
cp /opt/polymarket-analytics/deploy/systemd/*.service /etc/systemd/system/
cp /opt/polymarket-analytics/deploy/systemd/*.timer /etc/systemd/system/
systemctl daemon-reload

# Create data directories
echo "[8/8] Creating data directories..."
sudo -u polymarket mkdir -p /opt/polymarket-analytics/data
sudo -u polymarket mkdir -p /opt/polymarket-analytics/logs

# Setup logrotate
cat > /etc/logrotate.d/polymarket <<EOF
/var/log/polymarket/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 polymarket polymarket
}
EOF

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your .env file to /opt/polymarket-analytics/.env"
echo "   scp .env root@your-vps:/opt/polymarket-analytics/.env"
echo ""
echo "2. Set proper permissions:"
echo "   chown polymarket:polymarket /opt/polymarket-analytics/.env"
echo "   chmod 600 /opt/polymarket-analytics/.env"
echo ""
echo "3. Enable and start services:"
echo "   systemctl enable --now bond-bot"
echo "   systemctl enable --now flow-bot"
echo "   systemctl enable --now stat-arb-bot"
echo "   systemctl enable --now orderbook-recorder"
echo "   systemctl enable --now risk-monitor.timer"
echo ""
echo "4. Check status:"
echo "   systemctl status bond-bot flow-bot stat-arb-bot orderbook-recorder"
echo ""
