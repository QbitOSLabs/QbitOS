#!/usr/bin/env bash
# QbitOS Health Check Script
# Usage: ./deploy/healthcheck.sh [URL]

URL="${1:-http://localhost:8080/api/v1/health}"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null || echo "000")

if [ "$RESPONSE" = "200" ]; then
    echo "✓ QbitOS is healthy ($URL → 200)"
    exit 0
else
    echo "✗ QbitOS is not responding ($URL → $RESPONSE)"
    exit 1
fi
