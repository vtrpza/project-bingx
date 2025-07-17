"""
Asynchronous Rate Limiter
=========================

Provides a flexible async rate limiter to control the frequency of operations,
such as API calls.
"""

import asyncio
import time
from collections import deque

class AsyncRateLimiter:
    """
    An asynchronous rate limiter that restricts the number of calls made in a
    given time period.
    """

    def __init__(self, max_calls: int, period: float):
        """
        Initializes the rate limiter.

        Args:
            max_calls (int): The maximum number of calls allowed within the period.
            period (float): The time period in seconds.
        """
        self.max_calls = max_calls
        self.period = period
        self.semaphore = asyncio.Semaphore(max_calls)
        self.timestamps = deque()

    async def __aenter__(self):
        """Acquires a permit from the rate limiter."""
        await self.semaphore.acquire()
        now = time.monotonic()

        # Clean up old timestamps
        while self.timestamps and self.timestamps[0] <= now - self.period:
            self.timestamps.popleft()

        # If we are at the limit, calculate sleep time
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.timestamps[0] - (now - self.period)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.timestamps.append(time.monotonic())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Releases the permit."""
        self.semaphore.release()
