"""
Unit Tests for RecencySpace
============================

Standalone tests for the RecencySpace and TimeUnit classes.
No database connection required — tests only the encoding logic.

Run with:
    python -m pytest test/test_recency_space.py -v
"""

import math
import time
from datetime import datetime, timezone

import pytest

from pgvectordb.spaces import RecencySpace, TimeUnit

# ==================== TimeUnit Tests ====================


class TestTimeUnit:
    """Tests for the TimeUnit enum."""

    def test_second_to_seconds(self):
        assert TimeUnit.SECOND.to_seconds() == 1.0

    def test_minute_to_seconds(self):
        assert TimeUnit.MINUTE.to_seconds() == 60.0

    def test_hour_to_seconds(self):
        assert TimeUnit.HOUR.to_seconds() == 3600.0

    def test_day_to_seconds(self):
        assert TimeUnit.DAY.to_seconds() == 86400.0

    def test_week_to_seconds(self):
        assert TimeUnit.WEEK.to_seconds() == 604800.0

    def test_string_construction(self):
        """TimeUnit can be constructed from string values."""
        assert TimeUnit("day") == TimeUnit.DAY
        assert TimeUnit("hour") == TimeUnit.HOUR


# ==================== RecencySpace Init Tests ====================


class TestRecencySpaceInit:
    """Tests for RecencySpace initialization."""

    def test_valid_init(self):
        space = RecencySpace(
            name="created",
            field="created_at",
            time_unit=TimeUnit.DAY,
            period_value=7,
        )
        assert space.name == "created"
        assert space.field == "created_at"
        assert space.time_unit == TimeUnit.DAY
        assert space.period_value == 7.0
        assert space.tau == 7 * 86400.0
        assert space.dimensions == 1

    def test_string_time_unit(self):
        """TimeUnit can be passed as a string."""
        space = RecencySpace(name="ts", field="ts", time_unit="hour", period_value=2)
        assert space.time_unit == TimeUnit.HOUR
        assert space.tau == 2 * 3600.0

    def test_custom_dimensions(self):
        space = RecencySpace(name="ts", field="ts", dimensions=3)
        assert space.dimensions == 3

    def test_negative_period_raises(self):
        with pytest.raises(ValueError, match="period_value must be > 0"):
            RecencySpace(name="ts", field="ts", period_value=-1)

    def test_zero_period_raises(self):
        with pytest.raises(ValueError, match="period_value must be > 0"):
            RecencySpace(name="ts", field="ts", period_value=0)

    def test_zero_dimensions_raises(self):
        with pytest.raises(ValueError, match="dimensions must be >= 1"):
            RecencySpace(name="ts", field="ts", dimensions=0)

    def test_column_name(self):
        space = RecencySpace(name="updated", field="updated_at")
        assert space.column_name == "embedding_updated"

    def test_repr(self):
        space = RecencySpace(name="ts", field="ts", dimensions=2)
        assert "RecencySpace" in repr(space)
        assert "ts" in repr(space)
        assert "2" in repr(space)


# ==================== RecencySpace Encoding Tests ====================


class TestRecencySpaceEncode:
    """Tests for RecencySpace.encode()."""

    def test_encode_now_scores_near_one(self):
        """A timestamp from right now should score very close to 1.0."""
        space = RecencySpace(name="ts", field="ts", time_unit=TimeUnit.DAY)
        now_epoch = time.time()
        result = space.encode(now_epoch)
        assert len(result) == 1
        assert result[0] > 0.99

    def test_encode_old_timestamp_scores_near_zero(self):
        """A timestamp 30 days ago with τ=1 day should be near 0."""
        space = RecencySpace(name="ts", field="ts", time_unit=TimeUnit.DAY, period_value=1)
        old_epoch = time.time() - 30 * 86400  # 30 days ago
        result = space.encode(old_epoch)
        assert result[0] < 0.001

    def test_encode_one_tau_ago(self):
        """A timestamp exactly τ seconds ago should score ~0.368 (1/e)."""
        space = RecencySpace(
            name="ts",
            field="ts",
            time_unit=TimeUnit.HOUR,
            period_value=1,
        )
        one_tau_ago = time.time() - 3600  # 1 hour ago, τ = 3600s
        result = space.encode(one_tau_ago)
        expected = math.exp(-1)  # ≈ 0.3679
        assert abs(result[0] - expected) < 0.01

    def test_encode_future_timestamp_clamped(self):
        """Future timestamps should clamp to 1.0."""
        space = RecencySpace(name="ts", field="ts")
        future = time.time() + 100000
        result = space.encode(future)
        assert result[0] == 1.0

    def test_encode_none_returns_midpoint(self):
        """None value should encode as 0.5 (neutral)."""
        space = RecencySpace(name="ts", field="ts")
        result = space.encode(None)
        assert result == [0.5]

    def test_encode_iso_string(self):
        """ISO-8601 string should be parsed correctly."""
        space = RecencySpace(name="ts", field="ts", time_unit=TimeUnit.DAY, period_value=365)
        # A timestamp from now should be near 1.0
        now_iso = datetime.now(timezone.utc).isoformat()
        result = space.encode(now_iso)
        assert result[0] > 0.99

    def test_encode_iso_string_with_z(self):
        """ISO-8601 with 'Z' suffix should parse correctly."""
        space = RecencySpace(name="ts", field="ts", time_unit=TimeUnit.WEEK, period_value=52)
        result = space.encode("2026-02-19T12:00:00Z")
        assert 0.0 <= result[0] <= 1.0

    def test_encode_datetime_object(self):
        """datetime object should be handled correctly."""
        space = RecencySpace(name="ts", field="ts", time_unit=TimeUnit.DAY, period_value=365)
        now_dt = datetime.now(timezone.utc)
        result = space.encode(now_dt)
        assert result[0] > 0.99

    def test_encode_multi_dimensions(self):
        """Multi-dimensional RecencySpace should repeat the score."""
        space = RecencySpace(name="ts", field="ts", dimensions=3)
        result = space.encode(time.time())
        assert len(result) == 3
        assert all(v == result[0] for v in result)

    def test_invalid_timestamp_raises(self):
        """Invalid timestamp format should raise ValueError."""
        space = RecencySpace(name="ts", field="ts")
        with pytest.raises(ValueError, match="cannot parse timestamp"):
            space.encode("not-a-timestamp")

    def test_encode_integer_timestamp(self):
        """Integer timestamps should work."""
        space = RecencySpace(name="ts", field="ts", time_unit=TimeUnit.DAY, period_value=365)
        result = space.encode(int(time.time()))
        assert result[0] > 0.99


# ==================== RecencySpace Query Encoding Tests ====================


class TestRecencySpaceEncodeQuery:
    """Tests for RecencySpace.encode_query()."""

    def test_query_none_returns_one(self):
        """Default query (None) should return [1.0] — prefer newest."""
        space = RecencySpace(name="ts", field="ts")
        result = space.encode_query(None)
        assert result == [1.0]

    def test_query_none_multi_dims(self):
        """Default query with multiple dimensions."""
        space = RecencySpace(name="ts", field="ts", dimensions=4)
        result = space.encode_query(None)
        assert result == [1.0, 1.0, 1.0, 1.0]

    def test_query_with_timestamp(self):
        """Providing a timestamp to encode_query should behave like encode."""
        space = RecencySpace(name="ts", field="ts", time_unit=TimeUnit.DAY)
        ts = time.time() - 86400  # 1 day ago
        encode_result = space.encode(ts)
        query_result = space.encode_query(ts)
        assert abs(encode_result[0] - query_result[0]) < 0.001


# ==================== Timestamp Parsing Tests ====================


class TestTimestampParsing:
    """Tests for RecencySpace._to_epoch static method."""

    def test_int_passthrough(self):
        assert RecencySpace._to_epoch(1000000) == 1000000.0

    def test_float_passthrough(self):
        assert RecencySpace._to_epoch(1000000.5) == 1000000.5

    def test_datetime_utc(self):
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        epoch = RecencySpace._to_epoch(dt)
        assert isinstance(epoch, float)
        assert epoch > 0

    def test_datetime_naive(self):
        """Naive datetime should still convert (using system timezone)."""
        dt = datetime(2025, 6, 15, 12, 0, 0)
        epoch = RecencySpace._to_epoch(dt)
        assert isinstance(epoch, float)

    def test_iso_string(self):
        epoch = RecencySpace._to_epoch("2025-01-01T00:00:00+00:00")
        assert isinstance(epoch, float)

    def test_iso_string_z_suffix(self):
        epoch = RecencySpace._to_epoch("2025-01-01T00:00:00Z")
        assert isinstance(epoch, float)

    def test_numeric_string(self):
        epoch = RecencySpace._to_epoch("1000000.5")
        assert epoch == 1000000.5

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            RecencySpace._to_epoch("hello world")

    def test_list_raises(self):
        with pytest.raises(ValueError):
            RecencySpace._to_epoch([1, 2, 3])
