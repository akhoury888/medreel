"""Render worker:  python worker.py   (run one per CPU-heavy machine / GPU; set WORKER_INLINE=0 on the web app)."""
import logging, signal, threading
from saas.models import init_db
from saas.jobs import worker_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
if __name__ == "__main__":
    init_db()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    worker_loop(stop)
