#!/bin/bash
# Polymarket Bot Control Script
# Usage: ./polyctl.sh [command] [service]

SERVICES="bond-bot flow-bot stat-arb-bot orderbook-recorder"

usage() {
    echo "Polymarket Bot Control"
    echo ""
    echo "Usage: $0 <command> [service]"
    echo ""
    echo "Commands:"
    echo "  status      Show status of all bots"
    echo "  start       Start all bots (or specific bot)"
    echo "  stop        Stop all bots (or specific bot)"
    echo "  restart     Restart all bots (or specific bot)"
    echo "  logs        Show live logs for a bot"
    echo "  enable      Enable bot to start on boot"
    echo "  disable     Disable bot from starting on boot"
    echo "  sync        Run risk monitor sync"
    echo "  risk        Show risk status"
    echo ""
    echo "Services: bond-bot, flow-bot, stat-arb-bot, orderbook-recorder"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 start flow-bot"
    echo "  $0 logs stat-arb-bot"
    echo "  $0 restart"
}

status() {
    echo "=== Bot Status ==="
    for svc in $SERVICES; do
        status=$(systemctl is-active $svc 2>/dev/null || echo "unknown")
        enabled=$(systemctl is-enabled $svc 2>/dev/null || echo "unknown")
        printf "%-20s %s (enabled: %s)\n" "$svc:" "$status" "$enabled"
    done
    echo ""
    echo "=== Risk Monitor Timer ==="
    systemctl status risk-monitor.timer --no-pager 2>/dev/null | head -5
}

start_service() {
    if [ -z "$1" ]; then
        echo "Starting all bots..."
        for svc in $SERVICES; do
            systemctl start $svc && echo "Started $svc" || echo "Failed to start $svc"
        done
    else
        systemctl start $1 && echo "Started $1" || echo "Failed to start $1"
    fi
}

stop_service() {
    if [ -z "$1" ]; then
        echo "Stopping all bots..."
        for svc in $SERVICES; do
            systemctl stop $svc && echo "Stopped $svc" || echo "Failed to stop $svc"
        done
    else
        systemctl stop $1 && echo "Stopped $1" || echo "Failed to stop $1"
    fi
}

restart_service() {
    if [ -z "$1" ]; then
        echo "Restarting all bots..."
        for svc in $SERVICES; do
            systemctl restart $svc && echo "Restarted $svc" || echo "Failed to restart $svc"
        done
    else
        systemctl restart $1 && echo "Restarted $1" || echo "Failed to restart $1"
    fi
}

show_logs() {
    if [ -z "$1" ]; then
        echo "Please specify a service. Available: $SERVICES"
        exit 1
    fi
    journalctl -u $1 -f
}

enable_service() {
    if [ -z "$1" ]; then
        echo "Enabling all bots..."
        for svc in $SERVICES; do
            systemctl enable $svc && echo "Enabled $svc"
        done
        systemctl enable risk-monitor.timer && echo "Enabled risk-monitor.timer"
    else
        systemctl enable $1 && echo "Enabled $1"
    fi
}

disable_service() {
    if [ -z "$1" ]; then
        echo "Disabling all bots..."
        for svc in $SERVICES; do
            systemctl disable $svc && echo "Disabled $svc"
        done
    else
        systemctl disable $1 && echo "Disabled $1"
    fi
}

run_sync() {
    echo "Running risk monitor sync..."
    cd /opt/polymarket-analytics
    sudo -u polymarket /opt/polymarket-analytics/venv/bin/python scripts/risk_monitor.py sync
}

show_risk() {
    echo "=== Risk Status ==="
    cd /opt/polymarket-analytics
    sudo -u polymarket /opt/polymarket-analytics/venv/bin/python scripts/risk_monitor.py status
    echo ""
    echo "=== Positions ==="
    sudo -u polymarket /opt/polymarket-analytics/venv/bin/python scripts/risk_monitor.py positions
}

case "$1" in
    status)
        status
        ;;
    start)
        start_service $2
        ;;
    stop)
        stop_service $2
        ;;
    restart)
        restart_service $2
        ;;
    logs)
        show_logs $2
        ;;
    enable)
        enable_service $2
        ;;
    disable)
        disable_service $2
        ;;
    sync)
        run_sync
        ;;
    risk)
        show_risk
        ;;
    *)
        usage
        exit 1
        ;;
esac
