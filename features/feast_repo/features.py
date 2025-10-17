from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, Bytes
from feast.value_type import ValueType

image = Entity(name='image_id', join_keys=['image_id'], value_type=ValueType.INT64)

batch_src = FileSource(
    path='../../artifacts/features.parquet',
    timestamp_field="updated_at",
)

mnist_features = FeatureView(
    name='mnist_stats',
    entities=[image],
    schema=[
        Field(name='flat', dtype=Bytes),
        Field(name='pix_mean', dtype=Float32),
        Field(name='pix_var', dtype=Float32),
        *[Field(name=f'hist_{i}', dtype=Float32) for i in range(16)],
        Field(name='label', dtype=Int64),
    ],
    source=batch_src,
    ttl=None,
)
