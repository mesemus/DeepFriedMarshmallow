def is_overridden(instance_func, class_func):
    # type: (MethodType, MethodType) -> bool
    return instance_func.__func__ is not class_func


def has_overriden_serialization_method(is_serializing, instance, base_class):
    # Returns True if the instance method is overridden for the given context (serializing or deserializing)
    if is_serializing:
        return is_overridden(instance._serialize, base_class._serialize)
    return is_overridden(instance._deserialize, base_class._deserialize)


def is_schema_overridden(schema: "marshmallow.Schema") -> bool:
    return hasattr(schema, "_is_jit") and schema._is_jit
