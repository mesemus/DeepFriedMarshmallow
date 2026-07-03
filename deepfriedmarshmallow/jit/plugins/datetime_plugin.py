"""Builtin inliner for marshmallow.fields.Date, DateTime, and Time (ISO)."""

from __future__ import annotations

import datetime as _dt
from contextlib import suppress

from deepfriedmarshmallow.compat import has_overriden_serialization_method

from . import register_builtin_field_inliner_factory


def _jit_naive_datetime(value):
    """Parse a naive datetime string; raise ValueError for tz-aware input.

    Raising ValueError triggers the JIT wrapper fallback, which lets marshmallow
    produce the exact ValidationError with the correct message.  Valid naive
    inputs are handled entirely on the fast JIT path.
    """
    if value is None:
        return None
    result = _dt.datetime.fromisoformat(value)
    if result.tzinfo is not None:
        error_message = f"tz-aware datetime rejected for NaiveDateTime: {value=}"
        raise ValueError(error_message)
    return result


def _jit_aware_datetime(value):
    """Parse an aware datetime string; raise ValueError for naive input.

    Same fallback contract as _jit_naive_datetime.
    """
    if value is None:
        return None
    result = _dt.datetime.fromisoformat(value)
    if result.tzinfo is None or result.tzinfo.utcoffset(result) is None:
        error_message = f"naive datetime rejected for AwareDateTime: {value=}"
        raise ValueError(error_message)
    return result


def _datetime_inliner_factory(field_obj, context) -> str | tuple | None:  # pragma: no cover
    try:
        from marshmallow import fields
    except Exception:
        return None

    if isinstance(field_obj, fields.Date):
        if has_overriden_serialization_method(context.is_serializing, field_obj, fields.Date):
            return None
        if context.is_serializing:
            return "({0}.isoformat() if {0} is not None else None)"
        # Load path: date.fromisoformat
        return ("datetime.date.fromisoformat({0}) if {0} is not None else None", "datetime")

    if isinstance(field_obj, fields.NaiveDateTime):
        if has_overriden_serialization_method(context.is_serializing, field_obj, fields.NaiveDateTime):
            return None
        if context.is_serializing:
            return "({0}.isoformat() if {0} is not None else None)"
        # Inject helper; raises ValueError for tz-aware strings so the JIT
        # wrapper falls back to marshmallow's _deserialize for the exact error.
        context.namespace["_jit_naive_datetime"] = _jit_naive_datetime
        return "_jit_naive_datetime({0})"

    if isinstance(field_obj, fields.AwareDateTime):
        if has_overriden_serialization_method(context.is_serializing, field_obj, fields.AwareDateTime):
            return None
        if context.is_serializing:
            return "({0}.isoformat() if {0} is not None else None)"
        # Same pattern as NaiveDateTime.
        context.namespace["_jit_aware_datetime"] = _jit_aware_datetime
        return "_jit_aware_datetime({0})"

    if isinstance(field_obj, fields.DateTime):
        if has_overriden_serialization_method(context.is_serializing, field_obj, fields.DateTime):
            return None
        if context.is_serializing:
            return "({0}.isoformat() if {0} is not None else None)"
        # Load path: datetime.fromisoformat
        return ("datetime.datetime.fromisoformat({0}) if {0} is not None else None", "datetime")

    if isinstance(field_obj, fields.Time):
        if has_overriden_serialization_method(context.is_serializing, field_obj, fields.Time):
            return None
        if context.is_serializing:
            return "({0}.isoformat() if {0} is not None else None)"
        # Load path: time.fromisoformat
        return ("datetime.time.fromisoformat({0}) if {0} is not None else None", "datetime")

    return None


def _register() -> None:
    register_builtin_field_inliner_factory(_datetime_inliner_factory)


with suppress(Exception):
    _register()
