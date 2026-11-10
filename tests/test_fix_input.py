"""Test file for auto-fix. Run with --fix and inspect the result."""
import logging

logger = logging.getLogger(__name__)

# LSI001 — should get extra={"statement_id": "..."} added
logger.info("User logged in")
logger.debug("Cache miss", exc_info=True)
logging.warning("Disk usage high")

# LSI003 — should get "statement_id" inserted into existing dict
logger.error("Connection failed", extra={"host": "db.prod", "retries": 3})
logger.info("Request complete", extra={"duration_ms": 120})
logger.warning("Slow query", extra={})

# LSI005 — should have the bad UUID replaced
logger.info("Something happened", extra={"statement_id": "not-a-uuid"})
logger.debug("Another thing", extra={"statement_id": "too-short"})

# LSI002 — NOT auto-fixable (extra is a variable)
ctx = {"statement_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
logger.info("Variable extra", extra=ctx)

# LSI004 — NOT auto-fixable (statement_id is a function call)
logger.info("Dynamic UUID", extra={"statement_id": generate_id()})

# Already valid — should NOT be touched
logger.info("All good", extra={"statement_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"})

# Multiline call — LSI001
logger.error(
    "Something went wrong: %s",
    error_message,
    exc_info=True,
)

# self.logger pattern — LSI001
class MyService:
    def run(self):
        self.logger.info("starting up")

# Receiver-name variants — LSI001
LOG.critical("upper-case module logger")
self._log.debug("private attribute")
audit_logger.warning("snake_case prefix")
auditLogger.warning("camelCase prefix")
logging.getLogger(__name__).info("direct getLogger call")
logger.log(logging.INFO, "explicit level")

# Bound to a logger in this file, so matched despite the name — LSI001
audit = logging.getLogger("audit")
audit.info("oddly named logger")

# Not logging calls — should NOT be touched
retry_state.outcome.exception()
future.exception(timeout=5)
math.log(x)
np.log(values)
parser.error("bad argument")
console.info("not a logger")
catalog.info("name merely ends in 'log'")
logger.exception()

# Suppressed — should NOT be touched
logger.info("Suppressed by code")  # noqa: LSI001
logger.info("Suppressed by blanket")  # noqa
logger.info(
    "Suppressed multi-line",
    exc_info=True,
)  # noqa: LSI001

# noqa for another code does not suppress — LSI001
logger.info("Other code")  # noqa: E501
