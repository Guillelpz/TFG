#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Apply database migrations
echo "➡️ Running migrations..."
python manage.py migrate

# Collect static files
echo "➡️ Collecting static files..."
python manage.py collectstatic --noinput
#Create media directories 
mkdir -p /app/media/outputs

# Start the server
echo "✅ Starting Gunicorn server..."
exec "$@"
