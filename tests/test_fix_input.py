"""Test file for auto-fix. Run with --fix and inspect the result."""
import logging

logger = logging.getLogger(__name__)

# LOG001 — should get extra={"statement_id": "..."} added
logger.info("User logged in")
logger.debug("Cache miss", exc_info=True)
logging.warning("Disk usage high")

# LOG003 — should get "statement_id" inserted into existing dict
logger.error("Connection failed", extra={"host": "db.prod", "retries": 3})
logger.info("Request complete", extra={"duration_ms": 120})
logger.warning("Slow query", extra={})

# LOG005 — should have the bad UUID replaced
logger.info("Something happened", extra={"statement_id": "not-a-uuid"})
logger.debug("Another thing", extra={"statement_id": "too-short"})

# LOG002 — NOT auto-fixable (extra is a variable)
ctx = {"statement_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
logger.info("Variable extra", extra=ctx)

# LOG004 — NOT auto-fixable (statement_id is a function call)
logger.info("Dynamic UUID", extra={"statement_id": generate_id()})

# Already valid — should NOT be touched
logger.info("All good", extra={"statement_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"})

# Multiline call — LOG001
logger.error(
    "Something went wrong: %s",
    error_message,
    exc_info=True,
)

# self.logger pattern — LOG001
class MyService:
    def run(self):
        self.logger.info("starting up")
