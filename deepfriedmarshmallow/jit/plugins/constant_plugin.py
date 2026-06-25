"""Builtin inliner for marshmallow.fields.Constant."""

from __future__ import annotations

from contextlib import suppress

from deepfriedmarshmallow.compat import has_overriden_serialization_method

from . import register_builtin_field_inliner_factory


def _constant_inliner_factory(field_obj, context) -> str | tuple | None:  # pragma: no cover
    try:
        from marshmallow import fields
    except Exception:
        return None

    if not isinstance(field_obj, fields.Constant):
        return None

    if has_overriden_serialization_method(context.is_serializing, field_obj, fields.Constant):
        return None

    suffix = str(id(field_obj))
    sym = f"__dfm_const_{suffix}"
    context.namespace[sym] = getattr(
        field_obj, "constant", getattr(field_obj, "_value", getattr(field_obj, "value", None))
    )
    if context.is_serializing:
        return sym

    # Deserialize path: return the constant for any non-None input, None otherwise.
    #
    # For None input + allow_none=False: produces None in res, which the DFM-generated
    # None-result check catches → raises ValueError → JIT fallback → marshmallow raises
    # the correct ValidationError("Field may not be null.").
    #
    # For None input + allow_none=True: marshmallow short-circuits None without calling
    # _deserialize, returning None as the field value.  Returning None here matches that.
    return f"({sym} if {{0}} is not None else None)"


def _register() -> None:
    register_builtin_field_inliner_factory(_constant_inliner_factory)


with suppress(Exception):
    _register()
