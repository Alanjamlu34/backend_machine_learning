#!/bin/bash
gunicorn -w 4 -b 0.0.0.0:${PORT:-6000} app:app