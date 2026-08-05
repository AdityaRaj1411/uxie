"""
Vault File Watcher — monitors the vault directory for .md file changes.

Uses watchdog to detect file create/modify/delete events, with debouncing
to avoid rapid re-triggers on editor saves (many editors write multiple times).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Awaitable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# Files to ignore (config files that shouldn't be indexed as knowledge)
_IGNORE_FILES = {"_buddy_config.md"}

# Debounce window in seconds — ignore rapid duplicate events
_DEBOUNCE_SECONDS = 0.5


class _VaultEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler with debouncing.

    Collects file change events and fires callbacks after a debounce window,
    ensuring we don't re-ingest a file multiple times during a single save.
    """

    def __init__(
        self,
        on_file_changed: Callable[[Path], Awaitable[None]],
        on_file_deleted: Callable[[Path], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__()
        self._on_file_changed = on_file_changed
        self._on_file_deleted = on_file_deleted
        self._loop = loop
        self._pending: dict[str, tuple[str, float]] = {}  # path -> (event_type, timestamp)
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _should_handle(self, path: str) -> bool:
        """Check if this file should trigger re-indexing."""
        p = Path(path)
        supported_exts = {".md", ".txt", ".pdf", ".json", ".csv"}
        return (
            p.suffix.lower() in supported_exts
            and p.name not in _IGNORE_FILES
            and not p.name.startswith(".")
        )

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._schedule(event.src_path, "changed")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._schedule(event.src_path, "changed")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._schedule(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        # Treat move as delete old + create new
        if not event.is_directory:
            if self._should_handle(event.src_path):
                self._schedule(event.src_path, "deleted")
            if hasattr(event, "dest_path") and self._should_handle(event.dest_path):
                self._schedule(event.dest_path, "changed")

    def _schedule(self, path: str, event_type: str) -> None:
        """Schedule a debounced callback for a file event."""
        with self._lock:
            self._pending[path] = (event_type, time.monotonic())

            # Reset the debounce timer
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(_DEBOUNCE_SECONDS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Process all pending events after the debounce window."""
        with self._lock:
            events = dict(self._pending)
            self._pending.clear()
            self._timer = None

        for path, (event_type, _timestamp) in events.items():
            file_path = Path(path)
            if event_type == "deleted":
                logger.info("Watcher: file deleted → %s", file_path.name)
                asyncio.run_coroutine_threadsafe(
                    self._on_file_deleted(file_path), self._loop
                )
            else:
                logger.info("Watcher: file changed → %s", file_path.name)
                asyncio.run_coroutine_threadsafe(
                    self._on_file_changed(file_path), self._loop
                )


class VaultWatcher:
    """
    Filesystem watcher for the vault directory.

    Monitors for .md file changes and triggers re-indexing callbacks.
    Runs the watchdog Observer in a background thread.
    """

    def __init__(
        self,
        vault_path: Path | str,
        on_file_changed: Callable[[Path], Awaitable[None]],
        on_file_deleted: Callable[[Path], Awaitable[None]],
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self._vault_path = Path(vault_path)
        self._loop = loop or asyncio.get_event_loop()
        self._handler = _VaultEventHandler(on_file_changed, on_file_deleted, self._loop)
        self._observer = Observer()
        self._running = False

    def start(self) -> None:
        """Start watching the vault directory in a background thread."""
        if self._running:
            logger.warning("Watcher is already running.")
            return

        if not self._vault_path.exists():
            logger.warning("Vault path does not exist: %s", self._vault_path)
            return

        self._observer.schedule(self._handler, str(self._vault_path), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info("Vault watcher started: monitoring %s", self._vault_path)

    def stop(self) -> None:
        """Stop the watcher and clean up."""
        if not self._running:
            return

        self._observer.stop()
        self._observer.join(timeout=5)
        self._running = False
        logger.info("Vault watcher stopped.")

    @property
    def is_running(self) -> bool:
        return self._running
