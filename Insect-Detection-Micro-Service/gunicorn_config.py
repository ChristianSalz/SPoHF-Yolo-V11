import multiprocessing

# Server socket
bind = "0.0.0.0:7733"

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "insect_detection_api"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Restart workers after this many requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50