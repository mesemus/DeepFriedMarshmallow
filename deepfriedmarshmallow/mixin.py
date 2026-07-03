from deepfriedmarshmallow.serializer import JitDeserialize, JitSerialize


class JitSchemaMixin:
    jit_serialize_class = JitSerialize
    jit_deserialize_class = JitDeserialize
    _is_jit = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # note: neither jit_options not dfm are automatically copied from Meta to opts
        # (see marshmallow's SchemaOpts class) so we copy them manually
        meta = getattr(self, "Meta", None)
        if meta is not None:
            self.opts.jit_options = getattr(meta, "jit_options", {})
            self.opts.dfm = getattr(meta, "dfm", {})

        self._serialize = self.jit_serialize_class(self)
        self._deserialize = self.jit_deserialize_class(self)
