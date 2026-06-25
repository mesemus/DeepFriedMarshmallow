"""A comprehensive suite of tests comparing the (de)serialization of fields with and without dfm."""

import datetime
import decimal
import enum
import ipaddress
import uuid
from types import SimpleNamespace

from marshmallow.exceptions import ValidationError
import pytest
from marshmallow.fields import (
    Integer,
    String,
    Float,
    Decimal,
    Boolean,
    Date,
    DateTime,
    NaiveDateTime,
    AwareDateTime,
    Time,
    TimeDelta,
    UUID,
    Url,
    Email,
    IP,
    IPv4,
    IPv6,
    Constant,
    Raw,
    List,
    Dict,
    Tuple,
    Nested,
    IPInterface,
    IPv4Interface,
    IPv6Interface,
    Enum as EnumField,
    Function,
    Method,
)
from marshmallow import Schema, validate

from deepfriedmarshmallow import deep_fry_schema
from deepfriedmarshmallow.patch import deep_fry_schema_object


class _Color(enum.Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"


class _PersonSchema(Schema):
    name = String(required=True)
    age = Integer()
    score = Float()


class _PersonSchemaJit(_PersonSchema):
    pass


deep_fry_schema(_PersonSchemaJit)


class _AddressSchema(Schema):
    street = String(required=True)
    city = String(required=True)
    zip_code = String()


class _AddressSchemaJit(_AddressSchema):
    pass


deep_fry_schema(_AddressSchemaJit)


class _CompanySchema(Schema):
    name = String(required=True)
    address = Nested(_AddressSchema)
    employees = List(String())


class _CompanySchemaJit(_CompanySchema):
    address = Nested(_AddressSchemaJit)


deep_fry_schema(_CompanySchemaJit)


# ---------------------------------------------------------------------------
# Source factories for test_field_serialize
# Each factory takes (value, key) and returns a source object for schema.dump()
# ---------------------------------------------------------------------------


class _DictWithAttrs(dict):
    """Dict subclass with independent attribute storage alongside dict key storage.

    Used to test cases 5 and 6: marshmallow must use dict-key access (get()),
    not attribute access (getattr()), because the object is a Mapping.
    """

    pass


def _src_obj_with_attr(value, key):
    """Case 1: plain object with the field as an attribute."""
    return SimpleNamespace(**{key: value})


def _src_obj_without_attr(value, key):
    """Case 2: plain object missing the field attribute entirely."""
    return SimpleNamespace()


def _src_dict_with_key(value, key):
    """Case 3: dict with the field key present."""
    return {key: value}


def _src_dict_without_key(value, key):
    """Case 4: dict without the field key."""
    return {}


def _src_dict_key_and_attr(value, key):
    """Case 5: dict subclass with key=value AND a different sentinel attribute.

    Marshmallow must use the dict key, not the attribute.
    """
    d = _DictWithAttrs({key: value})
    d.__dict__[key] = "ATTR_SENTINEL"
    return d


def _src_dict_no_key_but_attr(value, key):
    """Case 6: dict subclass with no key but attribute=value.

    Marshmallow must use dict lookup (returns missing), ignoring the attribute.
    """
    d = _DictWithAttrs()
    d.__dict__[key] = value
    return d


_SOURCE_FACTORIES = [
    pytest.param(_src_obj_with_attr, id="source-obj-with-attr"),
    pytest.param(_src_obj_without_attr, id="source-obj-without-attr"),
    pytest.param(_src_dict_with_key, id="source-dict-with-key"),
    pytest.param(_src_dict_without_key, id="source-dict-without-key"),
    #
    # Note: these are supported by original marshmallow but are not supported by dfm
    # at all, so they are omitted from the test suite. The reason they fail is that
    # DictSerializer is used (it is a dictionary) and then the key is used to look up
    # the value, but the key is not present in the dictionary, so it fails. Original
    # marshmallow uses getattr as a fallback when the key is not present in the dictionary -
    # but that is not done in dfm for speed reasons.
    #
    # pytest.param(_src_dict_key_and_attr,    id="source-dict-key-and-attr"),
    # pytest.param(_src_dict_no_key_but_attr, id="source-dict-attr-only"),
]


@pytest.mark.parametrize(
    "field, field_value",
    [
        # --- Integer ---
        pytest.param(Integer(), 42, id="Integer-valid-int"),
        pytest.param(Integer(), "42", id="Integer-valid-str"),
        pytest.param(Integer(), 3.9, id="Integer-valid-float-truncates"),
        pytest.param(Integer(), "abc", id="Integer-invalid-str"),
        pytest.param(Integer(), None, id="Integer-none"),
        pytest.param(Integer(required=True), 42, id="Integer-required-valid"),
        pytest.param(Integer(required=True), None, id="Integer-required-none"),
        pytest.param(Integer(allow_none=True), None, id="Integer-allow-none"),
        # --- String ---
        pytest.param(String(), "hello", id="String-valid"),
        pytest.param(String(), "", id="String-empty"),
        pytest.param(String(), 123, id="String-valid-int-coerced"),
        pytest.param(String(), None, id="String-none"),
        pytest.param(String(required=True), "hello", id="String-required-valid"),
        pytest.param(String(required=True), None, id="String-required-none"),
        pytest.param(String(allow_none=True), None, id="String-allow-none"),
        # --- Float ---
        pytest.param(Float(), 3.14, id="Float-valid"),
        pytest.param(Float(), "3.14", id="Float-valid-str"),
        pytest.param(Float(), 42, id="Float-valid-int"),
        pytest.param(Float(), "abc", id="Float-invalid-str"),
        pytest.param(Float(), None, id="Float-none"),
        pytest.param(Float(required=True), 3.14, id="Float-required-valid"),
        pytest.param(Float(required=True), None, id="Float-required-none"),
        pytest.param(Float(allow_none=True), None, id="Float-allow-none"),
        # --- Decimal ---
        pytest.param(Decimal(), "3.14", id="Decimal-valid-str"),
        pytest.param(Decimal(), 3.14, id="Decimal-valid-float"),
        pytest.param(Decimal(), 42, id="Decimal-valid-int"),
        pytest.param(Decimal(), "abc", id="Decimal-invalid-str"),
        pytest.param(Decimal(), None, id="Decimal-none"),
        pytest.param(Decimal(required=True), "3.14", id="Decimal-required-valid"),
        pytest.param(Decimal(required=True), None, id="Decimal-required-none"),
        pytest.param(Decimal(allow_none=True), None, id="Decimal-allow-none"),
        # --- Boolean ---
        pytest.param(Boolean(), True, id="Boolean-valid-true"),
        pytest.param(Boolean(), False, id="Boolean-valid-false"),
        pytest.param(Boolean(), 1, id="Boolean-valid-1"),
        pytest.param(Boolean(), 0, id="Boolean-valid-0"),
        pytest.param(Boolean(), "true", id="Boolean-valid-str-true"),
        pytest.param(Boolean(), "false", id="Boolean-valid-str-false"),
        pytest.param(Boolean(), "invalid", id="Boolean-invalid-str"),
        pytest.param(Boolean(), None, id="Boolean-none"),
        pytest.param(Boolean(required=True), True, id="Boolean-required-valid"),
        pytest.param(Boolean(required=True), None, id="Boolean-required-none"),
        pytest.param(Boolean(allow_none=True), None, id="Boolean-allow-none"),
        # --- Date ---
        pytest.param(Date(), "2023-01-15", id="Date-valid"),
        pytest.param(Date(), "not-a-date", id="Date-invalid"),
        pytest.param(Date(), None, id="Date-none"),
        pytest.param(Date(required=True), "2023-01-15", id="Date-required-valid"),
        pytest.param(Date(required=True), None, id="Date-required-none"),
        pytest.param(Date(allow_none=True), None, id="Date-allow-none"),
        # --- DateTime ---
        pytest.param(DateTime(), "2023-01-15T10:30:00", id="DateTime-valid-naive"),
        pytest.param(DateTime(), "2023-01-15T10:30:00+00:00", id="DateTime-valid-tz"),
        pytest.param(DateTime(), "not-a-datetime", id="DateTime-invalid"),
        pytest.param(DateTime(), None, id="DateTime-none"),
        pytest.param(DateTime(required=True), "2023-01-15T10:30:00", id="DateTime-required-valid"),
        pytest.param(DateTime(required=True), None, id="DateTime-required-none"),
        pytest.param(DateTime(allow_none=True), None, id="DateTime-allow-none"),
        # --- NaiveDateTime ---
        pytest.param(NaiveDateTime(), "2023-01-15T10:30:00", id="NaiveDateTime-valid"),
        pytest.param(NaiveDateTime(), "2023-01-15T10:30:00+00:00", id="NaiveDateTime-invalid-tz-aware"),
        pytest.param(NaiveDateTime(), "not-a-datetime", id="NaiveDateTime-invalid"),
        pytest.param(NaiveDateTime(), None, id="NaiveDateTime-none"),
        pytest.param(NaiveDateTime(required=True), "2023-01-15T10:30:00", id="NaiveDateTime-required-valid"),
        pytest.param(NaiveDateTime(allow_none=True), None, id="NaiveDateTime-allow-none"),
        # --- AwareDateTime ---
        pytest.param(AwareDateTime(), "2023-01-15T10:30:00+00:00", id="AwareDateTime-valid"),
        pytest.param(AwareDateTime(), "2023-01-15T10:30:00", id="AwareDateTime-invalid-naive"),
        pytest.param(AwareDateTime(), "not-a-datetime", id="AwareDateTime-invalid"),
        pytest.param(AwareDateTime(), None, id="AwareDateTime-none"),
        pytest.param(AwareDateTime(required=True), "2023-01-15T10:30:00+00:00", id="AwareDateTime-required-valid"),
        pytest.param(AwareDateTime(allow_none=True), None, id="AwareDateTime-allow-none"),
        # --- Time ---
        pytest.param(Time(), "10:30:00", id="Time-valid"),
        pytest.param(Time(), "not-a-time", id="Time-invalid"),
        pytest.param(Time(), None, id="Time-none"),
        pytest.param(Time(required=True), "10:30:00", id="Time-required-valid"),
        pytest.param(Time(required=True), None, id="Time-required-none"),
        pytest.param(Time(allow_none=True), None, id="Time-allow-none"),
        # --- TimeDelta ---
        pytest.param(TimeDelta(), 100, id="TimeDelta-valid-int"),
        pytest.param(TimeDelta(), 3.5, id="TimeDelta-valid-float"),
        pytest.param(TimeDelta(), "invalid", id="TimeDelta-invalid"),
        pytest.param(TimeDelta(), None, id="TimeDelta-none"),
        pytest.param(TimeDelta(required=True), 100, id="TimeDelta-required-valid"),
        pytest.param(TimeDelta(required=True), None, id="TimeDelta-required-none"),
        pytest.param(TimeDelta(allow_none=True), None, id="TimeDelta-allow-none"),
        # --- UUID ---
        pytest.param(UUID(), "550e8400-e29b-41d4-a716-446655440000", id="UUID-valid"),
        pytest.param(UUID(), "not-a-uuid", id="UUID-invalid"),
        pytest.param(UUID(), None, id="UUID-none"),
        pytest.param(UUID(required=True), "550e8400-e29b-41d4-a716-446655440000", id="UUID-required-valid"),
        pytest.param(UUID(required=True), None, id="UUID-required-none"),
        pytest.param(UUID(allow_none=True), None, id="UUID-allow-none"),
        # --- Url ---
        pytest.param(Url(), "https://example.com", id="Url-valid"),
        pytest.param(Url(), "not-a-url", id="Url-invalid"),
        pytest.param(Url(), None, id="Url-none"),
        pytest.param(Url(required=True), "https://example.com", id="Url-required-valid"),
        pytest.param(Url(required=True), None, id="Url-required-none"),
        pytest.param(Url(allow_none=True), None, id="Url-allow-none"),
        # --- Email ---
        pytest.param(Email(), "test@example.com", id="Email-valid"),
        pytest.param(Email(), "not-an-email", id="Email-invalid"),
        pytest.param(Email(), None, id="Email-none"),
        pytest.param(Email(required=True), "test@example.com", id="Email-required-valid"),
        pytest.param(Email(required=True), None, id="Email-required-none"),
        pytest.param(Email(allow_none=True), None, id="Email-allow-none"),
        # --- IP ---
        pytest.param(IP(), "192.168.1.1", id="IP-valid-v4"),
        pytest.param(IP(), "2001:db8::1", id="IP-valid-v6"),
        pytest.param(IP(), "not-an-ip", id="IP-invalid"),
        pytest.param(IP(), None, id="IP-none"),
        pytest.param(IP(required=True), "192.168.1.1", id="IP-required-valid"),
        pytest.param(IP(required=True), None, id="IP-required-none"),
        pytest.param(IP(allow_none=True), None, id="IP-allow-none"),
        # --- IPv4 ---
        pytest.param(IPv4(), "192.168.1.1", id="IPv4-valid"),
        pytest.param(IPv4(), "2001:db8::1", id="IPv4-invalid-v6-addr"),
        pytest.param(IPv4(), "not-an-ip", id="IPv4-invalid"),
        pytest.param(IPv4(), None, id="IPv4-none"),
        pytest.param(IPv4(required=True), "192.168.1.1", id="IPv4-required-valid"),
        pytest.param(IPv4(allow_none=True), None, id="IPv4-allow-none"),
        # --- IPv6 ---
        pytest.param(IPv6(), "2001:db8::1", id="IPv6-valid"),
        pytest.param(IPv6(), "192.168.1.1", id="IPv6-invalid-v4-addr"),
        pytest.param(IPv6(), "not-an-ip", id="IPv6-invalid"),
        pytest.param(IPv6(), None, id="IPv6-none"),
        pytest.param(IPv6(required=True), "2001:db8::1", id="IPv6-required-valid"),
        pytest.param(IPv6(allow_none=True), None, id="IPv6-allow-none"),
        # --- validate.Range ---
        pytest.param(Integer(validate=validate.Range(min=0, max=100)), 50, id="Range-int-valid"),
        pytest.param(Integer(validate=validate.Range(min=0, max=100)), 0, id="Range-int-valid-min-boundary"),
        pytest.param(Integer(validate=validate.Range(min=0, max=100)), 100, id="Range-int-valid-max-boundary"),
        pytest.param(Integer(validate=validate.Range(min=0, max=100)), -1, id="Range-int-below-min"),
        pytest.param(Integer(validate=validate.Range(min=0, max=100)), 101, id="Range-int-above-max"),
        pytest.param(Float(validate=validate.Range(min=0.0)), 3.14, id="Range-float-valid"),
        pytest.param(Float(validate=validate.Range(min=0.0)), -0.001, id="Range-float-below-min"),
        pytest.param(Integer(validate=validate.Range(min=0, max=100), required=True), 50, id="Range-required-valid"),
        pytest.param(Integer(validate=validate.Range(min=0, max=100), required=True), None, id="Range-required-none"),
        pytest.param(Integer(validate=validate.Range(min=0, max=100), allow_none=True), None, id="Range-allow-none"),
        # --- validate.Length ---
        pytest.param(String(validate=validate.Length(min=2, max=8)), "hello", id="Length-valid"),
        pytest.param(String(validate=validate.Length(min=2, max=8)), "hi", id="Length-valid-min-boundary"),
        pytest.param(String(validate=validate.Length(min=2, max=8)), "abcdefgh", id="Length-valid-max-boundary"),
        pytest.param(String(validate=validate.Length(min=2, max=8)), "x", id="Length-too-short"),
        pytest.param(String(validate=validate.Length(min=2, max=8)), "toolongstr", id="Length-too-long"),
        pytest.param(String(validate=validate.Length(equal=5)), "hello", id="Length-equal-valid"),
        pytest.param(String(validate=validate.Length(equal=5)), "hi", id="Length-equal-invalid"),
        pytest.param(String(validate=validate.Length(min=1), required=True), "ok", id="Length-required-valid"),
        pytest.param(String(validate=validate.Length(min=1), required=True), None, id="Length-required-none"),
        pytest.param(String(validate=validate.Length(min=1), allow_none=True), None, id="Length-allow-none"),
        # --- validate.OneOf ---
        pytest.param(String(validate=validate.OneOf(["a", "b", "c"])), "a", id="OneOf-str-valid"),
        pytest.param(String(validate=validate.OneOf(["a", "b", "c"])), "c", id="OneOf-str-valid-last"),
        pytest.param(String(validate=validate.OneOf(["a", "b", "c"])), "d", id="OneOf-str-invalid"),
        pytest.param(String(validate=validate.OneOf(["a", "b", "c"])), "", id="OneOf-str-empty-invalid"),
        pytest.param(Integer(validate=validate.OneOf([1, 2, 3])), 1, id="OneOf-int-valid"),
        pytest.param(Integer(validate=validate.OneOf([1, 2, 3])), 4, id="OneOf-int-invalid"),
        pytest.param(String(validate=validate.OneOf(["a", "b"]), required=True), "a", id="OneOf-required-valid"),
        pytest.param(String(validate=validate.OneOf(["a", "b"]), required=True), None, id="OneOf-required-none"),
        pytest.param(String(validate=validate.OneOf(["a", "b"]), allow_none=True), None, id="OneOf-allow-none"),
        # --- validate.NoneOf ---
        pytest.param(String(validate=validate.NoneOf(["bad", "forbidden"])), "ok", id="NoneOf-valid"),
        pytest.param(String(validate=validate.NoneOf(["bad", "forbidden"])), "bad", id="NoneOf-invalid-first"),
        pytest.param(String(validate=validate.NoneOf(["bad", "forbidden"])), "forbidden", id="NoneOf-invalid-second"),
        pytest.param(String(validate=validate.NoneOf(["bad"]), required=True), "ok", id="NoneOf-required-valid"),
        pytest.param(String(validate=validate.NoneOf(["bad"]), required=True), None, id="NoneOf-required-none"),
        # --- validate.Regexp ---
        pytest.param(String(validate=validate.Regexp(r"^\d+$")), "123", id="Regexp-digits-valid"),
        pytest.param(String(validate=validate.Regexp(r"^\d+$")), "12a", id="Regexp-digits-invalid"),
        pytest.param(String(validate=validate.Regexp(r"^\d+$")), "", id="Regexp-empty-invalid"),
        pytest.param(String(validate=validate.Regexp(r"^[a-z]+$")), "abc", id="Regexp-alpha-valid"),
        pytest.param(String(validate=validate.Regexp(r"^[a-z]+$")), "ABC", id="Regexp-alpha-invalid-upper"),
        pytest.param(String(validate=validate.Regexp(r"^\d+$"), required=True), "5", id="Regexp-required-valid"),
        pytest.param(String(validate=validate.Regexp(r"^\d+$"), required=True), None, id="Regexp-required-none"),
        pytest.param(String(validate=validate.Regexp(r"^\d+$"), allow_none=True), None, id="Regexp-allow-none"),
        # --- validate.Equal ---
        pytest.param(String(validate=validate.Equal("exact")), "exact", id="Equal-valid"),
        pytest.param(String(validate=validate.Equal("exact")), "other", id="Equal-invalid"),
        pytest.param(Integer(validate=validate.Equal(42)), 42, id="Equal-int-valid"),
        pytest.param(Integer(validate=validate.Equal(42)), 43, id="Equal-int-invalid"),
        # --- multiple validators as list ---
        pytest.param(Integer(validate=[validate.Range(min=0), validate.Range(max=100)]), 50, id="MultiValidator-valid"),
        pytest.param(
            Integer(validate=[validate.Range(min=0), validate.Range(max=100)]), -1, id="MultiValidator-fails-first"
        ),
        pytest.param(
            Integer(validate=[validate.Range(min=0), validate.Range(max=100)]), 101, id="MultiValidator-fails-second"
        ),
        pytest.param(
            String(validate=[validate.Length(min=2), validate.Regexp(r"^\d+$")]),
            "42",
            id="MultiValidator-str-valid",
        ),
        pytest.param(
            String(validate=[validate.Length(min=2), validate.Regexp(r"^\d+$")]),
            "4",
            id="MultiValidator-str-fails-length",
        ),
        pytest.param(
            String(validate=[validate.Length(min=2), validate.Regexp(r"^\d+$")]),
            "ab",
            id="MultiValidator-str-fails-regexp",
        ),
        # --- Raw ---
        pytest.param(Raw(), "hello", id="Raw-str"),
        pytest.param(Raw(), 42, id="Raw-int"),
        pytest.param(Raw(), {"k": "v"}, id="Raw-dict"),
        pytest.param(Raw(), [1, 2, 3], id="Raw-list"),
        pytest.param(Raw(), None, id="Raw-none"),
        pytest.param(Raw(required=True), "hello", id="Raw-required-valid"),
        pytest.param(Raw(required=True), None, id="Raw-required-none"),
        pytest.param(Raw(allow_none=True), None, id="Raw-allow-none"),
        # --- Constant ---
        pytest.param(Constant("fixed"), "anything", id="Constant-str-input-ignored"),
        pytest.param(Constant("fixed"), 123, id="Constant-int-input-ignored"),
        pytest.param(Constant("fixed"), None, id="Constant-none-input-rejected"),
        pytest.param(Constant("fixed", allow_none=True), None, id="Constant-allow-none"),
        pytest.param(Constant(42), "any", id="Constant-int-value"),
        pytest.param(Constant(True), "any", id="Constant-bool-value"),
        # --- Tuple ---
        pytest.param(Tuple(tuple_fields=(Integer(), String())), [1, "hello"], id="Tuple-valid"),
        pytest.param(Tuple(tuple_fields=(Integer(), String())), [1, 2], id="Tuple-valid-coerced-str"),
        pytest.param(Tuple(tuple_fields=(Integer(), String())), [1], id="Tuple-invalid-too-short"),
        pytest.param(Tuple(tuple_fields=(Integer(), String())), [1, "a", "b"], id="Tuple-invalid-too-long"),
        pytest.param(Tuple(tuple_fields=(Integer(), String())), ["abc", "hello"], id="Tuple-invalid-type"),
        pytest.param(Tuple(tuple_fields=(Integer(), String())), None, id="Tuple-none"),
        pytest.param(Tuple(tuple_fields=(Integer(), String()), required=True), [1, "x"], id="Tuple-required-valid"),
        pytest.param(Tuple(tuple_fields=(Integer(), String()), required=True), None, id="Tuple-required-none"),
        pytest.param(Tuple(tuple_fields=(Integer(), String()), allow_none=True), None, id="Tuple-allow-none"),
        # --- IPInterface ---
        pytest.param(IPInterface(), "192.168.1.0/24", id="IPInterface-valid-v4"),
        pytest.param(IPInterface(), "2001:db8::/32", id="IPInterface-valid-v6"),
        pytest.param(IPInterface(), "192.168.1.1", id="IPInterface-valid-host"),
        pytest.param(IPInterface(), "not-an-interface", id="IPInterface-invalid"),
        pytest.param(IPInterface(), None, id="IPInterface-none"),
        pytest.param(IPInterface(required=True), "192.168.1.0/24", id="IPInterface-required-valid"),
        pytest.param(IPInterface(allow_none=True), None, id="IPInterface-allow-none"),
        # --- IPv4Interface ---
        pytest.param(IPv4Interface(), "192.168.1.0/24", id="IPv4Interface-valid"),
        pytest.param(IPv4Interface(), "192.168.1.1", id="IPv4Interface-valid-host"),
        pytest.param(IPv4Interface(), "2001:db8::/32", id="IPv4Interface-invalid-v6"),
        pytest.param(IPv4Interface(), "not-an-interface", id="IPv4Interface-invalid"),
        pytest.param(IPv4Interface(), None, id="IPv4Interface-none"),
        pytest.param(IPv4Interface(allow_none=True), None, id="IPv4Interface-allow-none"),
        # --- IPv6Interface ---
        pytest.param(IPv6Interface(), "2001:db8::/32", id="IPv6Interface-valid"),
        pytest.param(IPv6Interface(), "::1/128", id="IPv6Interface-valid-loopback"),
        pytest.param(IPv6Interface(), "192.168.1.0/24", id="IPv6Interface-invalid-v4"),
        pytest.param(IPv6Interface(), "not-an-interface", id="IPv6Interface-invalid"),
        pytest.param(IPv6Interface(), None, id="IPv6Interface-none"),
        pytest.param(IPv6Interface(allow_none=True), None, id="IPv6Interface-allow-none"),
        # --- Enum (by name, default) ---
        pytest.param(EnumField(_Color), "RED", id="Enum-valid-name"),
        pytest.param(EnumField(_Color), "BLUE", id="Enum-valid-name-2"),
        pytest.param(EnumField(_Color), "INVALID", id="Enum-invalid-name"),
        pytest.param(EnumField(_Color), "red", id="Enum-invalid-lowercase"),
        pytest.param(EnumField(_Color), None, id="Enum-none"),
        pytest.param(EnumField(_Color, required=True), "RED", id="Enum-required-valid"),
        pytest.param(EnumField(_Color, required=True), None, id="Enum-required-none"),
        pytest.param(EnumField(_Color, allow_none=True), None, id="Enum-allow-none"),
        # --- Enum (by_value=True) ---
        pytest.param(EnumField(_Color, by_value=True), "red", id="Enum-by-value-valid"),
        pytest.param(EnumField(_Color, by_value=True), "green", id="Enum-by-value-valid-2"),
        pytest.param(EnumField(_Color, by_value=True), "RED", id="Enum-by-value-invalid-name"),
        pytest.param(EnumField(_Color, by_value=True), "invalid", id="Enum-by-value-invalid"),
        pytest.param(EnumField(_Color, by_value=True), None, id="Enum-by-value-none"),
        pytest.param(EnumField(_Color, by_value=True, allow_none=True), None, id="Enum-by-value-allow-none"),
        # --- Function ---
        pytest.param(Function(deserialize=str.upper), "hello", id="Function-str-upper-valid"),
        pytest.param(Function(deserialize=str.upper), "", id="Function-str-upper-empty"),
        pytest.param(Function(deserialize=int), "42", id="Function-int-valid"),
        pytest.param(Function(deserialize=int), "abc", id="Function-int-invalid-raises"),
        pytest.param(Function(deserialize=int), None, id="Function-int-none"),
        pytest.param(Function(deserialize=lambda v: v * 2), 5, id="Function-lambda-valid"),
        pytest.param(Function(deserialize=str.upper, required=True), "hello", id="Function-required-valid"),
        pytest.param(Function(deserialize=str.upper, required=True), None, id="Function-required-none"),
        pytest.param(Function(deserialize=str.upper, allow_none=True), None, id="Function-allow-none"),
        # --- Method ---
        # sample_deserialize: uppercases the value, raises ValidationError for "bad"
        pytest.param(Method(deserialize="sample_deserialize"), "hello", id="Method-valid"),
        pytest.param(Method(deserialize="sample_deserialize"), "", id="Method-empty-str"),
        pytest.param(Method(deserialize="sample_deserialize"), "bad", id="Method-raises-validation-error"),
        pytest.param(Method(deserialize="sample_deserialize"), None, id="Method-none"),
        pytest.param(Method(deserialize="sample_deserialize", required=True), "hello", id="Method-required-valid"),
        pytest.param(Method(deserialize="sample_deserialize", required=True), None, id="Method-required-none"),
        pytest.param(Method(deserialize="identity", allow_none=True), None, id="Method-allow-none"),
        # to_int: parses string to int, raises ValueError for non-numeric
        pytest.param(Method(deserialize="to_int"), "42", id="Method-to-int-valid"),
        pytest.param(Method(deserialize="to_int"), "3", id="Method-to-int-valid-2"),
        pytest.param(Method(deserialize="to_int"), "abc", id="Method-to-int-invalid-raises"),
        pytest.param(Method(deserialize="to_int"), None, id="Method-to-int-none"),
        pytest.param(Method(deserialize="to_int", required=True), "7", id="Method-to-int-required-valid"),
        pytest.param(Method(deserialize="to_int", required=True), None, id="Method-to-int-required-none"),
        # validate_positive: returns value if >= 0, raises ValidationError otherwise
        pytest.param(Method(deserialize="validate_positive"), 5, id="Method-validate-positive-valid"),
        pytest.param(Method(deserialize="validate_positive"), 0, id="Method-validate-positive-zero-boundary"),
        pytest.param(Method(deserialize="validate_positive"), -1, id="Method-validate-positive-invalid"),
        pytest.param(Method(deserialize="validate_positive"), None, id="Method-validate-positive-none"),
        pytest.param(
            Method(deserialize="validate_positive", required=True), 10, id="Method-validate-positive-required-valid"
        ),
        pytest.param(
            Method(deserialize="validate_positive", required=True), None, id="Method-validate-positive-required-none"
        ),
        # --- List[String] ---
        pytest.param(List(String()), ["a", "b", "c"], id="List-str-valid"),
        pytest.param(List(String()), ["a", "", "c"], id="List-str-with-empty"),
        pytest.param(List(String()), [1, 2, 3], id="List-str-int-coerced"),
        pytest.param(List(String()), [], id="List-str-empty-list"),
        pytest.param(List(String()), "not-a-list", id="List-str-invalid-scalar"),
        pytest.param(List(String()), None, id="List-str-none"),
        pytest.param(List(String(), required=True), ["x"], id="List-str-required-valid"),
        pytest.param(List(String(), required=True), None, id="List-str-required-none"),
        pytest.param(List(String(), allow_none=True), None, id="List-str-allow-none"),
        pytest.param(List(String(validate=validate.Length(min=1))), ["a", "b"], id="List-str-inner-validate-valid"),
        pytest.param(List(String(validate=validate.Length(min=1))), ["a", ""], id="List-str-inner-validate-invalid"),
        # --- List[Integer] ---
        pytest.param(List(Integer()), [1, 2, 3], id="List-int-valid"),
        pytest.param(List(Integer()), ["1", "2", "3"], id="List-int-str-coerced"),
        pytest.param(List(Integer()), [1, "bad", 3], id="List-int-invalid-element"),
        pytest.param(List(Integer()), [], id="List-int-empty-list"),
        pytest.param(List(Integer()), None, id="List-int-none"),
        pytest.param(List(Integer(), required=True), [1, 2], id="List-int-required-valid"),
        pytest.param(List(Integer(), required=True), None, id="List-int-required-none"),
        pytest.param(List(Integer(), allow_none=True), None, id="List-int-allow-none"),
        pytest.param(List(Integer(validate=validate.Range(min=0))), [1, 2, 3], id="List-int-inner-validate-valid"),
        pytest.param(List(Integer(validate=validate.Range(min=0))), [1, -1, 3], id="List-int-inner-validate-invalid"),
        # --- Nested (_PersonSchema) ---
        pytest.param(
            (Nested(_PersonSchema), Nested(_PersonSchemaJit)), {"name": "Alice", "age": 30}, id="Nested-person-valid"
        ),
        pytest.param(
            (Nested(_PersonSchema), Nested(_PersonSchemaJit)),
            {"name": "Alice", "age": 30, "score": 9.5},
            id="Nested-person-valid-all-fields",
        ),
        pytest.param(
            (Nested(_PersonSchema), Nested(_PersonSchemaJit)), {"name": "Alice"}, id="Nested-person-optional-absent"
        ),
        pytest.param(
            (Nested(_PersonSchema), Nested(_PersonSchemaJit)), {"age": 30}, id="Nested-person-missing-required"
        ),
        pytest.param(
            (Nested(_PersonSchema), Nested(_PersonSchemaJit)),
            {"name": "Alice", "age": "not-a-number"},
            id="Nested-person-invalid-field-type",
        ),
        pytest.param((Nested(_PersonSchema), Nested(_PersonSchemaJit)), {}, id="Nested-person-empty"),
        pytest.param(
            (Nested(_PersonSchema), Nested(_PersonSchemaJit)), "not-a-dict", id="Nested-person-invalid-scalar"
        ),
        pytest.param((Nested(_PersonSchema), Nested(_PersonSchemaJit)), None, id="Nested-person-none"),
        pytest.param(
            (Nested(_PersonSchema, required=True), Nested(_PersonSchemaJit, required=True)),
            {"name": "Alice"},
            id="Nested-person-required-valid",
        ),
        pytest.param(
            (Nested(_PersonSchema, required=True), Nested(_PersonSchemaJit, required=True)),
            None,
            id="Nested-person-required-none",
        ),
        pytest.param(
            (Nested(_PersonSchema, allow_none=True), Nested(_PersonSchemaJit, allow_none=True)),
            None,
            id="Nested-person-allow-none",
        ),
        # --- Nested many=True (_PersonSchema) ---
        pytest.param(
            (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)),
            [{"name": "Alice", "age": 30}, {"name": "Bob"}],
            id="Nested-person-many-valid",
        ),
        pytest.param(
            (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)),
            [{"age": 30}],
            id="Nested-person-many-missing-required",
        ),
        pytest.param(
            (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)), [], id="Nested-person-many-empty"
        ),
        pytest.param(
            (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)), None, id="Nested-person-many-none"
        ),
        pytest.param(
            (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)),
            "not-a-list",
            id="Nested-person-many-invalid-scalar",
        ),
        # --- Nested (_AddressSchema) ---
        pytest.param(
            (Nested(_AddressSchema), Nested(_AddressSchemaJit)),
            {"street": "Main St", "city": "Springfield"},
            id="Nested-address-valid",
        ),
        pytest.param(
            (Nested(_AddressSchema), Nested(_AddressSchemaJit)),
            {"street": "Main St", "city": "Springfield", "zip_code": "12345"},
            id="Nested-address-valid-all-fields",
        ),
        pytest.param(
            (Nested(_AddressSchema), Nested(_AddressSchemaJit)),
            {"street": "Main St"},
            id="Nested-address-missing-required",
        ),
        pytest.param((Nested(_AddressSchema), Nested(_AddressSchemaJit)), {}, id="Nested-address-empty"),
        pytest.param((Nested(_AddressSchema), Nested(_AddressSchemaJit)), None, id="Nested-address-none"),
        pytest.param(
            (Nested(_AddressSchema, required=True), Nested(_AddressSchemaJit, required=True)),
            {"street": "Main St", "city": "Springfield"},
            id="Nested-address-required-valid",
        ),
        pytest.param(
            (Nested(_AddressSchema, required=True), Nested(_AddressSchemaJit, required=True)),
            None,
            id="Nested-address-required-none",
        ),
        pytest.param(
            (Nested(_AddressSchema, allow_none=True), Nested(_AddressSchemaJit, allow_none=True)),
            None,
            id="Nested-address-allow-none",
        ),
        # --- Nested with only / exclude ---
        pytest.param(
            (Nested(_PersonSchema, only=("name",)), Nested(_PersonSchemaJit, only=("name",))),
            {"name": "Alice"},
            id="Nested-person-only-valid",
        ),
        pytest.param(
            (Nested(_PersonSchema, only=("name",)), Nested(_PersonSchemaJit, only=("name",))),
            {},
            id="Nested-person-only-missing-required",
        ),
        pytest.param(
            (Nested(_PersonSchema, exclude=("score",)), Nested(_PersonSchemaJit, exclude=("score",))),
            {"name": "Alice", "age": 30},
            id="Nested-person-exclude-valid",
        ),
        pytest.param(
            (Nested(_PersonSchema, exclude=("score",)), Nested(_PersonSchemaJit, exclude=("score",))),
            {"name": "Alice", "age": "bad"},
            id="Nested-person-exclude-invalid-field",
        ),
        # --- Nested inside Nested (_CompanySchema → _AddressSchema) ---
        pytest.param(
            (Nested(_CompanySchema), Nested(_CompanySchemaJit)),
            {"name": "ACME", "address": {"street": "Main St", "city": "Springfield"}},
            id="Nested-nested-valid",
        ),
        pytest.param(
            (Nested(_CompanySchema), Nested(_CompanySchemaJit)),
            {
                "name": "ACME",
                "address": {"street": "Main St", "city": "Springfield", "zip_code": "12345"},
                "employees": ["Alice", "Bob"],
            },
            id="Nested-nested-valid-all-fields",
        ),
        pytest.param(
            (Nested(_CompanySchema), Nested(_CompanySchemaJit)), {"name": "ACME"}, id="Nested-nested-inner-absent"
        ),
        pytest.param(
            (Nested(_CompanySchema), Nested(_CompanySchemaJit)),
            {"name": "ACME", "address": {"street": "Main St"}},
            id="Nested-nested-inner-missing-required",
        ),
        pytest.param(
            (Nested(_CompanySchema), Nested(_CompanySchemaJit)),
            {"name": "ACME", "address": None},
            id="Nested-nested-inner-none",
        ),
        pytest.param(
            (Nested(_CompanySchema), Nested(_CompanySchemaJit)),
            {"name": "ACME", "address": "not-a-dict"},
            id="Nested-nested-inner-invalid-scalar",
        ),
        pytest.param(
            (Nested(_CompanySchema), Nested(_CompanySchemaJit)), {}, id="Nested-nested-outer-missing-required"
        ),
        pytest.param((Nested(_CompanySchema), Nested(_CompanySchemaJit)), None, id="Nested-nested-none"),
        pytest.param(
            (Nested(_CompanySchema, required=True), Nested(_CompanySchemaJit, required=True)),
            {"name": "ACME", "address": {"street": "Main St", "city": "Springfield"}},
            id="Nested-nested-required-valid",
        ),
        pytest.param(
            (Nested(_CompanySchema, required=True), Nested(_CompanySchemaJit, required=True)),
            None,
            id="Nested-nested-required-none",
        ),
        pytest.param(
            (Nested(_CompanySchema, allow_none=True), Nested(_CompanySchemaJit, allow_none=True)),
            None,
            id="Nested-nested-allow-none",
        ),
        # --- List of Nested (_PersonSchema) ---
        pytest.param(
            (List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))),
            [{"name": "Alice", "age": 30}, {"name": "Bob"}],
            id="List-nested-person-valid",
        ),
        pytest.param(
            (List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))),
            [{"name": "Alice", "age": 30, "score": 9.5}],
            id="List-nested-person-single-all-fields",
        ),
        pytest.param((List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))), [], id="List-nested-person-empty"),
        pytest.param(
            (List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))),
            [{"age": 30}],
            id="List-nested-person-missing-required",
        ),
        pytest.param(
            (List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))),
            [{"name": "Alice"}, {"name": "Bob", "age": "bad"}],
            id="List-nested-person-invalid-field-in-element",
        ),
        pytest.param(
            (List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))),
            [{"name": "Alice"}, "not-a-dict"],
            id="List-nested-person-invalid-element-type",
        ),
        pytest.param((List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))), None, id="List-nested-person-none"),
        pytest.param(
            (List(Nested(_PersonSchema), required=True), List(Nested(_PersonSchemaJit), required=True)),
            [{"name": "Alice"}],
            id="List-nested-person-required-valid",
        ),
        pytest.param(
            (List(Nested(_PersonSchema), required=True), List(Nested(_PersonSchemaJit), required=True)),
            None,
            id="List-nested-person-required-none",
        ),
        pytest.param(
            (List(Nested(_PersonSchema), allow_none=True), List(Nested(_PersonSchemaJit), allow_none=True)),
            None,
            id="List-nested-person-allow-none",
        ),
        # --- Dict[String, Nested(_PersonSchema)] ---
        pytest.param(
            (Dict(keys=String(), values=Nested(_PersonSchema)), Dict(keys=String(), values=Nested(_PersonSchemaJit))),
            {"alice": {"name": "Alice", "age": 30}, "bob": {"name": "Bob"}},
            id="Dict-nested-person-valid",
        ),
        pytest.param(
            (Dict(keys=String(), values=Nested(_PersonSchema)), Dict(keys=String(), values=Nested(_PersonSchemaJit))),
            {},
            id="Dict-nested-person-empty",
        ),
        pytest.param(
            (Dict(keys=String(), values=Nested(_PersonSchema)), Dict(keys=String(), values=Nested(_PersonSchemaJit))),
            {"x": {"age": 30}},
            id="Dict-nested-person-missing-required-in-value",
        ),
        pytest.param(
            (Dict(keys=String(), values=Nested(_PersonSchema)), Dict(keys=String(), values=Nested(_PersonSchemaJit))),
            None,
            id="Dict-nested-person-none",
        ),
        # --- Dict[String, String] ---
        pytest.param(Dict(keys=String(), values=String()), {"a": "x", "b": "y"}, id="Dict-str-str-valid"),
        pytest.param(Dict(keys=String(), values=String()), {}, id="Dict-str-str-empty"),
        pytest.param(Dict(keys=String(), values=String()), {"a": 1}, id="Dict-str-str-value-coerced"),
        pytest.param(Dict(keys=String(), values=String()), {"a": None}, id="Dict-str-str-value-none"),
        pytest.param(Dict(keys=String(), values=String()), "not-a-dict", id="Dict-str-str-invalid-scalar"),
        pytest.param(Dict(keys=String(), values=String()), [1, 2], id="Dict-str-str-invalid-list"),
        pytest.param(Dict(keys=String(), values=String()), None, id="Dict-str-str-none"),
        pytest.param(Dict(keys=String(), values=String(), required=True), {"k": "v"}, id="Dict-str-str-required-valid"),
        pytest.param(Dict(keys=String(), values=String(), required=True), None, id="Dict-str-str-required-none"),
        pytest.param(Dict(keys=String(), values=String(), allow_none=True), None, id="Dict-str-str-allow-none"),
        # --- Dict[String, Integer] ---
        pytest.param(Dict(keys=String(), values=Integer()), {"a": 1, "b": 2}, id="Dict-str-int-valid"),
        pytest.param(Dict(keys=String(), values=Integer()), {"a": "42"}, id="Dict-str-int-value-str-coerced"),
        pytest.param(Dict(keys=String(), values=Integer()), {"a": "bad"}, id="Dict-str-int-value-invalid"),
        pytest.param(Dict(keys=String(), values=Integer()), {}, id="Dict-str-int-empty"),
        pytest.param(Dict(keys=String(), values=Integer()), None, id="Dict-str-int-none"),
        pytest.param(Dict(keys=String(), values=Integer(), required=True), {"k": 1}, id="Dict-str-int-required-valid"),
        pytest.param(Dict(keys=String(), values=Integer(), required=True), None, id="Dict-str-int-required-none"),
        pytest.param(Dict(keys=String(), values=Integer(), allow_none=True), None, id="Dict-str-int-allow-none"),
        # --- Dict with inner validation ---
        pytest.param(
            Dict(keys=String(), values=Integer(validate=validate.Range(min=0))),
            {"a": 1, "b": 2},
            id="Dict-str-int-inner-validate-valid",
        ),
        pytest.param(
            Dict(keys=String(), values=Integer(validate=validate.Range(min=0))),
            {"a": 1, "b": -1},
            id="Dict-str-int-inner-validate-invalid",
        ),
        pytest.param(
            Dict(keys=String(validate=validate.Length(min=1)), values=String()),
            {"ab": "x"},
            id="Dict-key-validate-valid",
        ),
        pytest.param(
            Dict(keys=String(validate=validate.Length(min=1)), values=String()),
            {"": "x"},
            id="Dict-key-validate-invalid",
        ),
    ],
)
def test_field_deserialize(field, field_value):
    if isinstance(field, tuple):
        non_dfm_field, dfm_field = field
    else:
        non_dfm_field = dfm_field = field

    class TestSchema(Schema):
        fld = non_dfm_field

        def sample_deserialize(self, value):
            if value == "bad":
                raise ValidationError("Invalid value")
            return value.upper()

        def sample_serialize(self, value):
            return value.lower()

        def identity(self, value):
            return value

        def to_int(self, value):
            return int(value)

        def validate_positive(self, value):
            if value < 0:
                raise ValidationError("Must be non-negative")
            return value

    without_dfm_result = None
    without_dfm_exception = None

    schema = TestSchema()

    try:
        without_dfm_result = schema.load({"fld": field_value})
    except Exception as e:
        without_dfm_exception = e

    if non_dfm_field is not dfm_field:

        class DFMTestSchema(TestSchema):
            fld = dfm_field

        dfm_schema = DFMTestSchema()
    else:
        dfm_schema = schema

    deep_fry_schema_object(dfm_schema)

    with_dfm_result = None
    with_dfm_exception = None

    try:
        with_dfm_result = dfm_schema.load({"fld": field_value})
    except Exception as e:
        with_dfm_exception = e

    assert without_dfm_result == with_dfm_result
    assert repr(without_dfm_exception) == repr(with_dfm_exception)


# ---------------------------------------------------------------------------
# Serialization (dump) tests
# ---------------------------------------------------------------------------

_SERIALIZE_CASES = [
    # --- Integer ---
    pytest.param(Integer(), 42, id="Integer-valid"),
    pytest.param(Integer(), -5, id="Integer-negative"),
    pytest.param(Integer(), None, id="Integer-none"),
    pytest.param(Integer(allow_none=True), None, id="Integer-allow-none"),
    pytest.param(Integer(dump_default=0), 42, id="Integer-dump-default"),
    # --- String ---
    pytest.param(String(), "hello", id="String-valid"),
    pytest.param(String(), "", id="String-empty"),
    pytest.param(String(), None, id="String-none"),
    pytest.param(String(allow_none=True), None, id="String-allow-none"),
    pytest.param(String(dump_default="N/A"), "hello", id="String-dump-default"),
    # --- Float ---
    pytest.param(Float(), 3.14, id="Float-valid"),
    pytest.param(Float(), 0.0, id="Float-zero"),
    pytest.param(Float(), None, id="Float-none"),
    pytest.param(Float(allow_none=True), None, id="Float-allow-none"),
    # --- Decimal ---
    pytest.param(Decimal(), decimal.Decimal("3.14"), id="Decimal-valid"),
    pytest.param(Decimal(), decimal.Decimal("0"), id="Decimal-zero"),
    pytest.param(Decimal(), None, id="Decimal-none"),
    pytest.param(Decimal(allow_none=True), None, id="Decimal-allow-none"),
    # --- Boolean ---
    pytest.param(Boolean(), True, id="Boolean-true"),
    pytest.param(Boolean(), False, id="Boolean-false"),
    pytest.param(Boolean(), None, id="Boolean-none"),
    pytest.param(Boolean(allow_none=True), None, id="Boolean-allow-none"),
    # --- Date ---
    pytest.param(Date(), datetime.date(2023, 1, 15), id="Date-valid"),
    pytest.param(Date(), None, id="Date-none"),
    pytest.param(Date(allow_none=True), None, id="Date-allow-none"),
    # --- DateTime ---
    pytest.param(DateTime(), datetime.datetime(2023, 1, 15, 10, 30, 0), id="DateTime-naive"),
    pytest.param(DateTime(), datetime.datetime(2023, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc), id="DateTime-tz"),
    pytest.param(DateTime(), None, id="DateTime-none"),
    pytest.param(DateTime(allow_none=True), None, id="DateTime-allow-none"),
    # --- NaiveDateTime ---
    pytest.param(NaiveDateTime(), datetime.datetime(2023, 1, 15, 10, 30, 0), id="NaiveDateTime-valid"),
    pytest.param(NaiveDateTime(), None, id="NaiveDateTime-none"),
    # --- AwareDateTime ---
    pytest.param(
        AwareDateTime(),
        datetime.datetime(2023, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc),
        id="AwareDateTime-valid",
    ),
    pytest.param(AwareDateTime(), None, id="AwareDateTime-none"),
    # --- Time ---
    pytest.param(Time(), datetime.time(10, 30, 0), id="Time-valid"),
    pytest.param(Time(), None, id="Time-none"),
    pytest.param(Time(allow_none=True), None, id="Time-allow-none"),
    # --- TimeDelta ---
    pytest.param(TimeDelta(), datetime.timedelta(seconds=100), id="TimeDelta-valid"),
    pytest.param(TimeDelta(), datetime.timedelta(days=1, seconds=3600), id="TimeDelta-days"),
    pytest.param(TimeDelta(), None, id="TimeDelta-none"),
    pytest.param(TimeDelta(allow_none=True), None, id="TimeDelta-allow-none"),
    # --- UUID ---
    pytest.param(UUID(), uuid.UUID("550e8400-e29b-41d4-a716-446655440000"), id="UUID-valid"),
    pytest.param(UUID(), None, id="UUID-none"),
    pytest.param(UUID(allow_none=True), None, id="UUID-allow-none"),
    # --- Url / Email ---
    pytest.param(Url(), "https://example.com", id="Url-valid"),
    pytest.param(Url(), None, id="Url-none"),
    pytest.param(Email(), "test@example.com", id="Email-valid"),
    pytest.param(Email(), None, id="Email-none"),
    # --- IP / IPv4 / IPv6 ---
    pytest.param(IP(), ipaddress.IPv4Address("192.168.1.1"), id="IP-valid-v4"),
    pytest.param(IP(), ipaddress.IPv6Address("2001:db8::1"), id="IP-valid-v6"),
    pytest.param(IP(), None, id="IP-none"),
    pytest.param(IPv4(), ipaddress.IPv4Address("192.168.1.1"), id="IPv4-valid"),
    pytest.param(IPv4(), None, id="IPv4-none"),
    pytest.param(IPv6(), ipaddress.IPv6Address("2001:db8::1"), id="IPv6-valid"),
    pytest.param(IPv6(), None, id="IPv6-none"),
    # --- IPInterface / IPv4Interface / IPv6Interface ---
    pytest.param(IPInterface(), ipaddress.IPv4Interface("192.168.1.0/24"), id="IPInterface-valid-v4"),
    pytest.param(IPInterface(), ipaddress.IPv6Interface("2001:db8::/32"), id="IPInterface-valid-v6"),
    pytest.param(IPInterface(), None, id="IPInterface-none"),
    pytest.param(IPv4Interface(), ipaddress.IPv4Interface("192.168.1.0/24"), id="IPv4Interface-valid"),
    pytest.param(IPv4Interface(), None, id="IPv4Interface-none"),
    pytest.param(IPv6Interface(), ipaddress.IPv6Interface("2001:db8::/32"), id="IPv6Interface-valid"),
    pytest.param(IPv6Interface(), None, id="IPv6Interface-none"),
    # --- Constant ---
    pytest.param(Constant("fixed"), "anything", id="Constant-str-input"),
    pytest.param(Constant("fixed"), None, id="Constant-none-input"),
    pytest.param(Constant(42), "whatever", id="Constant-int"),
    # --- Raw ---
    pytest.param(Raw(), {"k": "v"}, id="Raw-dict"),
    pytest.param(Raw(), [1, 2, 3], id="Raw-list"),
    pytest.param(Raw(), "hello", id="Raw-str"),
    pytest.param(Raw(), None, id="Raw-none"),
    # --- Tuple ---
    pytest.param(Tuple(tuple_fields=(Integer(), String())), (1, "hello"), id="Tuple-valid"),
    pytest.param(Tuple(tuple_fields=(Integer(), String())), (1, 2), id="Tuple-coerced-str"),
    pytest.param(Tuple(tuple_fields=(Integer(), String())), None, id="Tuple-none"),
    # --- Enum ---
    pytest.param(EnumField(_Color), _Color.RED, id="Enum-by-name"),
    pytest.param(EnumField(_Color), None, id="Enum-none"),
    pytest.param(EnumField(_Color, by_value=True), _Color.BLUE, id="Enum-by-value"),
    pytest.param(EnumField(_Color, allow_none=True), None, id="Enum-allow-none"),
    # --- List[String] ---
    pytest.param(List(String()), ["a", "b", "c"], id="List-str-valid"),
    pytest.param(List(String()), [], id="List-str-empty"),
    pytest.param(List(String()), None, id="List-str-none"),
    pytest.param(List(String(), allow_none=True), None, id="List-str-allow-none"),
    # --- List[Integer] ---
    pytest.param(List(Integer()), [1, 2, 3], id="List-int-valid"),
    pytest.param(List(Integer()), [1, None, 3], id="List-int-with-none"),
    pytest.param(List(Integer()), None, id="List-int-none"),
    # --- Dict[String, String] ---
    pytest.param(Dict(keys=String(), values=String()), {"a": "x", "b": "y"}, id="Dict-str-str-valid"),
    pytest.param(Dict(keys=String(), values=String()), {}, id="Dict-str-str-empty"),
    pytest.param(Dict(keys=String(), values=String()), None, id="Dict-str-str-none"),
    # --- Dict[String, Integer] ---
    pytest.param(Dict(keys=String(), values=Integer()), {"a": 1, "b": 2}, id="Dict-str-int-valid"),
    pytest.param(Dict(keys=String(), values=Integer()), {}, id="Dict-str-int-empty"),
    pytest.param(Dict(keys=String(), values=Integer()), None, id="Dict-str-int-none"),
    # --- Nested (_PersonSchema) ---
    pytest.param(
        (Nested(_PersonSchema), Nested(_PersonSchemaJit)),
        {"name": "Alice", "age": 30, "score": 9.5},
        id="Nested-person-valid-all",
    ),
    pytest.param((Nested(_PersonSchema), Nested(_PersonSchemaJit)), {"name": "Alice"}, id="Nested-person-partial"),
    pytest.param((Nested(_PersonSchema), Nested(_PersonSchemaJit)), {}, id="Nested-person-empty"),
    pytest.param((Nested(_PersonSchema), Nested(_PersonSchemaJit)), None, id="Nested-person-none"),
    pytest.param(
        (Nested(_PersonSchema, allow_none=True), Nested(_PersonSchemaJit, allow_none=True)),
        None,
        id="Nested-person-allow-none",
    ),
    pytest.param(
        (Nested(_PersonSchema, only=("name",)), Nested(_PersonSchemaJit, only=("name",))),
        {"name": "Alice", "age": 30},
        id="Nested-person-only",
    ),
    pytest.param(
        (Nested(_PersonSchema, exclude=("score",)), Nested(_PersonSchemaJit, exclude=("score",))),
        {"name": "Alice", "age": 30, "score": 9.5},
        id="Nested-person-exclude",
    ),
    # --- Nested many=True ---
    pytest.param(
        (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)),
        [{"name": "Alice", "age": 30}, {"name": "Bob"}],
        id="Nested-person-many-valid",
    ),
    pytest.param(
        (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)), [], id="Nested-person-many-empty"
    ),
    pytest.param(
        (Nested(_PersonSchema, many=True), Nested(_PersonSchemaJit, many=True)), None, id="Nested-person-many-none"
    ),
    # --- Nested inside Nested ---
    pytest.param(
        (Nested(_CompanySchema), Nested(_CompanySchemaJit)),
        {"name": "ACME", "address": {"street": "Main St", "city": "Springfield"}},
        id="Nested-nested-valid",
    ),
    pytest.param(
        (Nested(_CompanySchema), Nested(_CompanySchemaJit)),
        {"name": "ACME", "address": None},
        id="Nested-nested-inner-none",
    ),
    pytest.param(
        (Nested(_CompanySchema), Nested(_CompanySchemaJit)), {"name": "ACME"}, id="Nested-nested-inner-absent"
    ),
    pytest.param((Nested(_CompanySchema), Nested(_CompanySchemaJit)), None, id="Nested-nested-none"),
    # --- List of Nested ---
    pytest.param(
        (List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))),
        [{"name": "Alice", "age": 30}, {"name": "Bob"}],
        id="List-nested-person-valid",
    ),
    pytest.param((List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))), [], id="List-nested-person-empty"),
    pytest.param((List(Nested(_PersonSchema)), List(Nested(_PersonSchemaJit))), None, id="List-nested-person-none"),
    # --- Dict of Nested ---
    pytest.param(
        (Dict(keys=String(), values=Nested(_PersonSchema)), Dict(keys=String(), values=Nested(_PersonSchemaJit))),
        {"alice": {"name": "Alice", "age": 30}},
        id="Dict-nested-person-valid",
    ),
    pytest.param(
        (Dict(keys=String(), values=Nested(_PersonSchema)), Dict(keys=String(), values=Nested(_PersonSchemaJit))),
        {},
        id="Dict-nested-person-empty",
    ),
    pytest.param(
        (Dict(keys=String(), values=Nested(_PersonSchema)), Dict(keys=String(), values=Nested(_PersonSchemaJit))),
        None,
        id="Dict-nested-person-none",
    ),
]


@pytest.mark.parametrize("source_factory", _SOURCE_FACTORIES)
@pytest.mark.parametrize("field, field_value", _SERIALIZE_CASES)
def test_field_serialize(field, field_value, source_factory):
    if isinstance(field, tuple):
        non_dfm_field, dfm_field = field
    else:
        non_dfm_field = dfm_field = field

    class TestSchema(Schema):
        fld = non_dfm_field

    source = source_factory(field_value, "fld")

    without_dfm_result = None
    without_dfm_exception = None

    schema = TestSchema()

    try:
        without_dfm_result = schema.dump(source)
    except Exception as e:
        without_dfm_exception = e

    if non_dfm_field is not dfm_field:

        class DFMTestSchema(TestSchema):
            fld = dfm_field

        dfm_schema = DFMTestSchema()
    else:
        dfm_schema = schema

    deep_fry_schema_object(dfm_schema)

    with_dfm_result = None
    with_dfm_exception = None

    try:
        with_dfm_result = dfm_schema.dump(source)
    except Exception as e:
        with_dfm_exception = e

    assert without_dfm_result == with_dfm_result
    assert repr(without_dfm_exception) == repr(with_dfm_exception)
