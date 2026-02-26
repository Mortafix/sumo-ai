import asyncio
from dataclasses import dataclass


@dataclass
class MetricsBucket:
    requests_total: int = 0
    success_total: int = 0
    failure_total: int = 0
    cache_hits_total: int = 0
    cache_misses_total: int = 0
    processing_total_ms: float = 0.0


class InMemoryMetrics:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._since_start = MetricsBucket()
        self._per_mode: dict[str, MetricsBucket] = {
            "one_line": MetricsBucket(),
            "veloce": MetricsBucket(),
            "dettagliato": MetricsBucket(),
        }

    async def record_request(self, mode: str) -> None:
        async with self._lock:
            self._since_start.requests_total += 1
            self._bucket_for_mode(mode).requests_total += 1

    async def record_success(
        self, mode: str, processing_ms: float, cached: bool
    ) -> None:
        async with self._lock:
            self._since_start.success_total += 1
            self._since_start.processing_total_ms += processing_ms
            mode_bucket = self._bucket_for_mode(mode)
            mode_bucket.success_total += 1
            mode_bucket.processing_total_ms += processing_ms

            if cached:
                self._since_start.cache_hits_total += 1
                mode_bucket.cache_hits_total += 1
            else:
                self._since_start.cache_misses_total += 1
                mode_bucket.cache_misses_total += 1

    async def record_failure(self, mode: str, processing_ms: float) -> None:
        async with self._lock:
            self._since_start.failure_total += 1
            self._since_start.processing_total_ms += processing_ms
            mode_bucket = self._bucket_for_mode(mode)
            mode_bucket.failure_total += 1
            mode_bucket.processing_total_ms += processing_ms

    async def snapshot(self) -> dict:
        async with self._lock:
            return {
                "since_start": self._bucket_snapshot(self._since_start),
                "per_mode": {
                    "one_line": self._bucket_snapshot(self._per_mode["one_line"]),
                    "veloce": self._bucket_snapshot(self._per_mode["veloce"]),
                    "dettagliato": self._bucket_snapshot(self._per_mode["dettagliato"]),
                },
            }

    def _bucket_for_mode(self, mode: str) -> MetricsBucket:
        candidate = (mode or "").strip().lower()
        if candidate == "one_line":
            return self._per_mode["one_line"]
        if candidate == "dettagliato":
            return self._per_mode["dettagliato"]
        return self._per_mode["veloce"]

    @staticmethod
    def _bucket_snapshot(bucket: MetricsBucket) -> dict:
        request_count = bucket.requests_total
        cache_events = bucket.cache_hits_total + bucket.cache_misses_total
        cache_hit_rate = (
            bucket.cache_hits_total / cache_events if cache_events > 0 else 0.0
        )
        error_rate = bucket.failure_total / request_count if request_count > 0 else 0.0
        avg_processing_ms = (
            bucket.processing_total_ms / request_count if request_count > 0 else 0.0
        )
        return {
            "requests_total": bucket.requests_total,
            "success_total": bucket.success_total,
            "failure_total": bucket.failure_total,
            "cache_hits_total": bucket.cache_hits_total,
            "cache_misses_total": bucket.cache_misses_total,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "error_rate": round(error_rate, 4),
            "avg_processing_ms": round(avg_processing_ms, 2),
        }
