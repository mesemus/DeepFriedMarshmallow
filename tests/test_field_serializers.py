from marshmallow import Schema

from deepfriedmarshmallow.jit import DictSerializer, FieldSerializer, JitContext, attr_str, generate_serialize_method, generate_transform_method_body
from deepfriedmarshmallow.utils import IndentedString
from deepfriedmarshmallow.jit.plugins import _registry

from marshmallow.fields import Int

class AttributeFieldSerializer(FieldSerializer):
    """Serializer that always serializes an attribute field."""
    def serialize(self, attr_name, field_symbol, assignment_template, field_obj):
        """Serializer that always serializes an attribute field."""
        # type: (str, str, str, marshmallow.fields.Field) -> IndentedString
        if self.context.is_serializing:
            return IndentedString(assignment_template.format(attr_str(attr_name)))

class AttributeInt(Int):
    """Int field that always serializes as an attribute. This is a mock only, a real field
    would be for example a marshmallow_utils.NestedAttribute field.

    (see https://github.com/inveniosoftware/marshmallow-utils/blob/master/marshmallow_utils/fields/nestedattr.py)
    """
    pass

class TestSchema(Schema):
    a_attribute = AttributeInt()
    a = Int()

class TestData(dict):
    def __init__(self, **kwargs):
        a_attribute = kwargs.pop("a_attribute", None)
        super().__init__(**kwargs)
        self.a_attribute = a_attribute

def test_attribute_field_serializer(monkeypatch):
    monkeypatch.setattr(_registry, "field_serializers", [(AttributeInt, AttributeFieldSerializer)])

    context = JitContext()
    context.is_serializing = True

    body = generate_transform_method_body(TestSchema(), DictSerializer(context), context)
    assert """
def DictSerializer(obj):
    res = {}
    value = obj.a_attribute; value = value() if callable(value) else value; res["a_attribute"] = _field_a_attribute__serialize(value, "a_attribute", obj)
    if "a" in obj:
        value = obj["a"]; value = value() if callable(value) else value; res["a"] = _field_a__serialize(value, "a", obj)
    return res
    """.strip() in str(body)

def test_jitted_schema(monkeypatch):
    monkeypatch.setattr(_registry, "field_serializers", [(AttributeInt, AttributeFieldSerializer)])
    serialize_method = generate_serialize_method(TestSchema())
    result = serialize_method(
        TestData(a_attribute=3, a=4)
    )
    expected = {
        "a_attribute": 3,
        "a": 4
    }
    assert expected == result
