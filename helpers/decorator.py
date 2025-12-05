import time
import os
from functools import wraps
from typing import Any, Callable, TypeVar, cast
from logger import LOG
import asyncio


T = TypeVar("T")


class TimeDecorator:
    """A decorator class for measuring and logging function execution time in development environment."""

    @classmethod
    def _log_execution_time(
        cls, func_name: str, start_time: float, error: str = None
    ) -> None:
        """
        Logs the execution time of a function with high precision.

        Args:
            func_name: Name of the function being measured
            start_time: Start time of function execution (from time.perf_counter)
            error: Optional error message if function execution failed
        """
        duration = time.perf_counter() - start_time
        log_message = f"Function '{func_name}' executed in {duration:.6f} seconds"
        if error:
            log_message += f" with error: {error}"
        LOG.debug(log_message)

    @classmethod
    def exe_time(cls, func: Callable[..., T]) -> Callable[..., T]:
        """Decorates a function to measure and log its execution time in the development environment.


            Args:
                func: The function to be decorated

            Returns:
                A wrapped function that includes timing measurement

            Raises:
                RuntimeError: If environment variable access fails
        """

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            """Wrapper for asynchronous functions."""
            try:
                if os.environ.get("PROJECT_ENV", "").lower() != "dev":
                    return await func(*args, **kwargs)

                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    cls._log_execution_time(func.__name__, start_time)
                    return result
                except Exception as e:
                    cls._log_execution_time(func.__name__, start_time, error=str(e))
                    raise
            except OSError as e:
                LOG.error(f"Failed to access environment variables: {e}")
                raise RuntimeError(f"Environment variable access failed: {e}")

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            """Wrapper for synchronous functions."""
            try:
                if os.environ.get("PROJECT_ENV", "").lower() != "dev":
                    return func(*args, **kwargs)

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    cls._log_execution_time(func.__name__, start_time)
                    return result
                except Exception as e:
                    cls._log_execution_time(func.__name__, start_time, error=str(e))
                    raise
            except OSError as e:
                LOG.error(f"Failed to access environment variables: {e}")
                raise RuntimeError(f"Environment variable access failed: {e}")

        return cast(
            Callable[..., T],
            async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper,
        )
