-- Create the test database alongside the default (ema_dev, set via POSTGRES_DB).
-- This script runs inside docker-entrypoint-initdb.d during first container startup.
CREATE DATABASE ema_test;
