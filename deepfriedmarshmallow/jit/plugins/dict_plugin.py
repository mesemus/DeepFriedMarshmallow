"""Builtin inliner for marshmallow.fields.Dict."""

from __future__ import annotations

from contextlib import suppress

from deepfriedmarshmallow.compat import has_overriden_serialization_method

from . import register_builtin_field_inliner_factory


def _inner_inliner(field_obj, context) -> str | tuple | None:
    """Return an inliner for a subfield or None if unsupported.

    Reuses core inliners and marshmallow-enum mapping.
    """
    from deepfriedmarshmallow.jit import (
        BooleanInliner,
        NumberInliner,
        StringInliner,
        UUIDInliner,
    )

    try:
        import marshmallow_enum as mm_enum  # type: ignore
    except Exception:  # pragma: no cover
        mm_enum = None  # type: ignore

    if mm_enum is not None and isinstance(field_obj, mm_enum.EnumField):
        enum_cls = getattr(field_obj, "enum", None)
        if enum_cls is None:
            return None
        mod = enum_cls.__module__
        name = enum_cls.__name__
        ref = f"{mod}.{name}" if mod else name
        if context.is_serializing:
            if getattr(field_obj, "by_value", False):
                return "({0}.value if {0} is not None else None)"
            return "({0}.name if {0} is not None else None)"
        if getattr(field_obj, "by_value", False):
            return (f"{ref}({{0}}) if {{0}} is not None else None", mod)
        return (f"{ref}[{{0}}] if {{0}} is not None else None", mod)

    for inliner_cls in (UUIDInliner, StringInliner, NumberInliner, BooleanInliner):
        try:
            result = inliner_cls().inline(field_obj, context)
        except Exception:
            result = None
        if result:
            return result
    return None


def _dict_inliner_factory(field_obj, context) -> str | tuple | None:
    try:
        from marshmallow import fields
    except Exception:
        return None

    if not isinstance(field_obj, fields.Dict):
        return None

    if has_overriden_serialization_method(context.is_serializing, field_obj, fields.Dict):
        return None

    # marshmallow 3 stores key/value fields as key_field / value_field
    key_field = getattr(field_obj, "key_field", None)
    val_field = getattr(field_obj, "value_field", None)

    # Identity mapping if subfield is not provided
    key_inline = None
    val_inline = None
    if key_field is not None:
        # If inner field has validators, fall back so they run
        if getattr(key_field, "validators", None):
            return None
        key_inline = _inner_inliner(key_field, context)
        if not key_inline:
            return None
    if val_field is not None:
        # If inner field has validators, fall back so they run
        if getattr(val_field, "validators", None):
            return None
        val_inline = _inner_inliner(val_field, context)
        if not val_inline:
            return None

    # Build expression and collect imports
    imports: list = []

    def _expr_or_identity(inliner, var):
        nonlocal imports
        if not inliner:
            return var
        if isinstance(inliner, tuple):
            expr, imps = inliner
            # normalize imports to list
            if isinstance(imps, list | set | tuple):
                imports.extend(list(imps))
            else:
                imports.append(imps)
        else:
            expr = inliner
        return expr.format(var)

    key_expr = _expr_or_identity(key_inline, "k")
    val_expr = _expr_or_identity(val_inline, "v")

    # For deserialization, inner fields that don't allow None must reject None explicitly.
    # StringInliner returns None for None input (relying on the outer JIT's None check),
    # but inside a dict comprehension there is no such outer guard.
    # The guard must come first so it isn't swallowed by the trailing ternary in the
    # StringInliner expression ("… else dict()['error']" has lower precedence).
    if not context.is_serializing:
        if key_field is not None and not getattr(key_field, "allow_none", False):
            key_expr = f"(dict()['error'] if k is None else ({key_expr}))"
        if val_field is not None and not getattr(val_field, "allow_none", False):
            val_expr = f"(dict()['error'] if v is None else ({val_expr}))"

    # Escape literal braces; leave {0} placeholders for JIT value substitution
    dict_expr = f"dict((({key_expr}, {val_expr}) for (k, v) in ({{0}}).items())) if {{0}} is not None else None"
    if imports:
        return (dict_expr, tuple(imports))
    return dict_expr


def _register() -> None:
    register_builtin_field_inliner_factory(_dict_inliner_factory)


with suppress(Exception):
    _register()
