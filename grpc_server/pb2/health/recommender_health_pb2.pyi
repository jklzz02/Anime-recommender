from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("anime_loader_status", "data_sets_status", "version", "service_status")
    ANIME_LOADER_STATUS_FIELD_NUMBER: _ClassVar[int]
    DATA_SETS_STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVICE_STATUS_FIELD_NUMBER: _ClassVar[int]
    anime_loader_status: AnimeLoaderStatus
    data_sets_status: DataSetStatus
    version: int
    service_status: str
    def __init__(self, anime_loader_status: _Optional[_Union[AnimeLoaderStatus, _Mapping]] = ..., data_sets_status: _Optional[_Union[DataSetStatus, _Mapping]] = ..., version: _Optional[int] = ..., service_status: _Optional[str] = ...) -> None: ...

class AnimeLoaderStatus(_message.Message):
    __slots__ = ("is_loaded", "has_error", "anime_count", "error_message", "cache_hits", "cache_misses", "cache_size", "cache_max_size")
    IS_LOADED_FIELD_NUMBER: _ClassVar[int]
    HAS_ERROR_FIELD_NUMBER: _ClassVar[int]
    ANIME_COUNT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CACHE_HITS_FIELD_NUMBER: _ClassVar[int]
    CACHE_MISSES_FIELD_NUMBER: _ClassVar[int]
    CACHE_SIZE_FIELD_NUMBER: _ClassVar[int]
    CACHE_MAX_SIZE_FIELD_NUMBER: _ClassVar[int]
    is_loaded: bool
    has_error: bool
    anime_count: int
    error_message: str
    cache_hits: int
    cache_misses: int
    cache_size: int
    cache_max_size: int
    def __init__(self, is_loaded: bool = ..., has_error: bool = ..., anime_count: _Optional[int] = ..., error_message: _Optional[str] = ..., cache_hits: _Optional[int] = ..., cache_misses: _Optional[int] = ..., cache_size: _Optional[int] = ..., cache_max_size: _Optional[int] = ...) -> None: ...

class DataSetStatus(_message.Message):
    __slots__ = ("is_healthy", "set_status")
    IS_HEALTHY_FIELD_NUMBER: _ClassVar[int]
    SET_STATUS_FIELD_NUMBER: _ClassVar[int]
    is_healthy: bool
    set_status: _containers.RepeatedCompositeFieldContainer[DataHealth]
    def __init__(self, is_healthy: bool = ..., set_status: _Optional[_Iterable[_Union[DataHealth, _Mapping]]] = ...) -> None: ...

class DataHealth(_message.Message):
    __slots__ = ("file", "status")
    FILE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    file: str
    status: str
    def __init__(self, file: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...
