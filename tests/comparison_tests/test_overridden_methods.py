"""Tests that DFM respects overridden _serialize / _deserialize on field subclasses.

Each custom class produces output that is observably different from its base
class, so any bypass of an override will cause a test failure.  The same schema
instance is run first without DFM, then patched with DFM; the results must be
identical.
"""

from collections import namedtuple
import datetime
import decimal
import uuid as _uuid_mod

import pytest
from marshmallow import Schema, fields

from deepfriedmarshmallow.jit import JitContext, generate_method_bodies, generate_serialize_method
from deepfriedmarshmallow.patch import deep_fry_schema_object

UUID_1_STR = "550e8400-e29b-41d4-a716-446655440000"
UUID_1_OBJ = _uuid_mod.UUID(UUID_1_STR)
UUID_2_STR = "660e8400-e29b-41d4-a716-446655440000"
UUID_2_OBJ = _uuid_mod.UUID(UUID_2_STR)
UUID_3_STR = "770e8400-e29b-41d4-a716-446655440000"
UUID_3_OBJ = _uuid_mod.UUID(UUID_3_STR)


# ---------------------------------------------------------------------------
# Custom field subclasses
# Each overrides both _serialize and _deserialize so the output differs from
# the base class in an easy-to-detect way.
# ---------------------------------------------------------------------------

ConstantSerializers = namedtuple("ConstantSerializers", ["serializer", "deserializer", "serializer_deserializer"])


def make_constant_fields(base_class, serializer_value, deserializer_value):
    class SerializationMixin:
        def _serialize(self, value, attr, obj, **kwargs):
            super()._serialize(value, attr, obj, **kwargs)
            return serializer_value

    class DeserializationMixin:
        def _deserialize(self, value, attr, data, **kwargs):
            super()._deserialize(value, attr, data, **kwargs)
            return deserializer_value

    return ConstantSerializers(
        type(
            f"{base_class.__name__}ConstantSerializerDeserializer",
            (SerializationMixin, DeserializationMixin, base_class),
            {},
        ),
        type(
            f"{base_class.__name__}ConstantSerializer",
            (
                SerializationMixin,
                base_class,
            ),
            {},
        ),
        type(
            f"{base_class.__name__}ConstantDeserializer",
            (
                DeserializationMixin,
                base_class,
            ),
            {},
        ),
    )


CONSTANT_STRINGS = make_constant_fields(fields.String, "foo", "bar")
CONSTANT_INTEGERS = make_constant_fields(fields.Integer, 42, 62)
CONSTANT_FLOATS = make_constant_fields(fields.Float, 3.14, 2.71)
CONSTANT_BOOLEANS = make_constant_fields(fields.Boolean, True, False)
CONSTANT_BOOLEANS = make_constant_fields(fields.Boolean, True, False)
CONSTANT_UUIDS = make_constant_fields(fields.UUID, UUID_1_OBJ, UUID_2_OBJ)
CONSTANT_DATES = make_constant_fields(fields.Date, datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
CONSTANT_DATETIMES = make_constant_fields(
    fields.DateTime, datetime.datetime(2023, 1, 1, 12, 23), datetime.datetime(2023, 12, 31, 11, 5)
)
CONSTANT_TIMEDELTAS = make_constant_fields(
    fields.TimeDelta, datetime.timedelta(seconds=3600), datetime.timedelta(days=1)
)
CONSTANT_DECIMALS = make_constant_fields(fields.Decimal, decimal.Decimal("3.14"), decimal.Decimal("2.71"))
CONSTANT_LISTS = make_constant_fields(fields.List, ["foo1", "foo2"], ["bar1", "bar2"])
CONSTANT_DICTS = make_constant_fields(fields.Dict, {"foo1": "bar1"}, {"foo2": "bar2"})
CONSTANT_NESTED = make_constant_fields(fields.Nested, {"field": "bar1"}, {"field": "bar2"})


class NestedSchema(Schema):
    field = fields.Str()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cmp(field, operation, value):
    """Run load/dump before and after DFM patching; assert outcomes are equal."""

    class S(Schema):
        fld = field

    schema = S()

    plain_result = plain_exc = None
    if operation == "load":
        plain_result = schema.load({"fld": value})
    else:
        plain_result = schema.dump({"fld": value})

    deep_fry_schema_object(schema)
    try:
        dfm_result = None
        if operation == "load":
            dfm_result = schema.load({"fld": value})
        else:
            dfm_result = schema.dump({"fld": value})

        assert plain_result == dfm_result
    except:
        ctx = JitContext()
        ctx.is_serializing = operation == "dump"
        print(generate_method_bodies(S(), ctx))
        raise


# ---------------------------------------------------------------------------
# Load (deserialize) cases
# ---------------------------------------------------------------------------

_LOAD_CASES = [
    # --- string ---
    pytest.param(CONSTANT_STRINGS, None, "hello", id="string-valid"),
    pytest.param(CONSTANT_INTEGERS, None, 5, id="int-valid"),
    pytest.param(CONSTANT_FLOATS, None, 5.0, id="float-valid"),
    pytest.param(CONSTANT_BOOLEANS, None, True, id="boolean-true"),
    pytest.param(CONSTANT_BOOLEANS, None, False, id="boolean-false"),
    pytest.param(CONSTANT_UUIDS, None, UUID_3_STR, id="uuid-valid"),
    pytest.param(CONSTANT_DATES, None, "1970-01-15", id="date-valid"),
    pytest.param(CONSTANT_DATETIMES, None, "1970-01-15T10:30:00", id="dateTime-naive"),
    pytest.param(CONSTANT_TIMEDELTAS, None, 2000, id="timedelta-int"),
    pytest.param(CONSTANT_DECIMALS, None, "-143", id="decimal-str"),
    # --- List with overridden inner field ---
    pytest.param(CONSTANT_STRINGS, lambda cls: fields.List(cls), ["hello", "world"], id="list-string-valid"),
    # list with its own serializer/deserializer
    pytest.param(CONSTANT_LISTS, lambda cls: cls(fields.Str), ["hello", "world"], id="list-constant-str"),
    pytest.param(
        CONSTANT_STRINGS,
        lambda cls: fields.Dict(keys=fields.String(), values=cls()),
        {"a": "hello", "b": "world"},
        id="dict-string-value-valid",
    ),
    pytest.param(
        CONSTANT_STRINGS,
        lambda cls: fields.Dict(keys=cls(), values=fields.String()),
        {"a": "hello", "b": "world"},
        id="dict-string-key-valid",
    ),
    pytest.param(
        CONSTANT_DICTS,
        lambda cls: cls(keys=fields.String(), values=fields.String()),
        {"a": "hello", "b": "world"},
        id="dict-constant-str",
    ),
    pytest.param(
        CONSTANT_NESTED,
        lambda cls: cls(NestedSchema),
        {"field": "blah"},
        id="nested-constant-str",
    ),
]


@pytest.mark.parametrize(
    "serializer_type",
    [
        "serializer",
        "deserializer",
        "serializer_deserializer",
    ],
)
@pytest.mark.parametrize("field, factory, value", _LOAD_CASES)
def test_overridden_field_load(serializer_type, field, factory, value):
    if not factory:
        factory = lambda cls: cls()
    _cmp(factory(getattr(field, serializer_type)), "load", value)


# ---------------------------------------------------------------------------
# Dump (serialize) cases
# ---------------------------------------------------------------------------


_DUMP_CASES = [
    # --- string ---
    pytest.param(CONSTANT_STRINGS, None, "hello", id="string-valid"),
    pytest.param(CONSTANT_INTEGERS, None, 5, id="int-valid"),
    pytest.param(CONSTANT_FLOATS, None, 5.0, id="float-valid"),
    pytest.param(CONSTANT_BOOLEANS, None, True, id="boolean-true"),
    pytest.param(CONSTANT_BOOLEANS, None, False, id="boolean-false"),
    pytest.param(CONSTANT_UUIDS, None, UUID_3_STR, id="uuid-valid"),
    pytest.param(CONSTANT_DATES, None, datetime.date.fromisoformat("1970-01-15"), id="date-valid"),
    pytest.param(CONSTANT_DATETIMES, None, datetime.datetime.fromisoformat("1970-01-15T10:30:00"), id="dateTime-naive"),
    pytest.param(CONSTANT_TIMEDELTAS, None, datetime.timedelta(seconds=8000), id="timedelta-int"),
    pytest.param(CONSTANT_DECIMALS, None, "-143", id="decimal-str"),
    # --- List with overridden inner field ---
    pytest.param(CONSTANT_STRINGS, lambda cls: fields.List(cls), ["hello", "world"], id="list-string-valid"),
    # list with its own serializer/deserializer
    pytest.param(CONSTANT_LISTS, lambda cls: cls(fields.Str), ["hello", "world"], id="list-constant-str"),
    pytest.param(
        CONSTANT_STRINGS,
        lambda cls: fields.Dict(keys=fields.String(), values=cls()),
        {"a": "hello", "b": "world"},
        id="dict-string-value-valid",
    ),
    pytest.param(
        CONSTANT_STRINGS,
        lambda cls: fields.Dict(keys=cls(), values=fields.String()),
        {"a": "hello", "b": "world"},
        id="dict-string-key-valid",
    ),
    pytest.param(
        CONSTANT_DICTS,
        lambda cls: cls(keys=fields.String(), values=fields.String()),
        {"a": "hello", "b": "world"},
        id="dict-constant-str",
    ),
    pytest.param(
        CONSTANT_NESTED,
        lambda cls: cls(NestedSchema),
        {"field": "blah"},
        id="nested-constant-str",
    ),
]


@pytest.mark.parametrize(
    "serializer_type",
    [
        "serializer",
        "deserializer",
        "serializer_deserializer",
    ],
)
@pytest.mark.parametrize("field, factory, value", _DUMP_CASES)
def test_overridden_field_dump(serializer_type, field, factory, value):
    if not factory:
        factory = lambda cls: cls()
    _cmp(factory(getattr(field, serializer_type)), "dump", value)
