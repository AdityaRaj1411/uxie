import functools
import logging

logger = logging.getLogger(__name__)

def safe_command(func):
    """Decorator to catch exceptions in slash commands."""
    @functools.wraps(func)
    async def wrapper(context, arguments: str):
        try:
            return await func(context, arguments)
        except Exception as e:
            logger.exception(f"Command execution failed: {e}")
            return f"Command error: {e}", context.session_manager.session.current_mode
    return wrapper

class CommandRegistry:
    def __init__(self):
        self._commands = {}

    def register(self, name: str):
        def decorator(func):
            self._commands[name.lower()] = func
            return func
        return decorator

    def get(self, name: str):
        return self._commands.get(name.lower())

registry = CommandRegistry()
