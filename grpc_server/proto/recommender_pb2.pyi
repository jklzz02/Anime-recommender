from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CompatibleRequest(_message.Message):
    __slots__ = ("user_favourite_ids", "limit")
    USER_FAVOURITE_IDS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    user_favourite_ids: _containers.RepeatedScalarFieldContainer[int]
    limit: int
    def __init__(self, user_favourite_ids: _Optional[_Iterable[int]] = ..., limit: _Optional[int] = ...) -> None: ...

class CompatibleAnime(_message.Message):
    __slots__ = ("anime_id", "compatibility_score")
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    anime_id: int
    compatibility_score: float
    def __init__(self, anime_id: _Optional[int] = ..., compatibility_score: _Optional[float] = ...) -> None: ...

class CompatibleResponse(_message.Message):
    __slots__ = ("data", "scale")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[CompatibleAnime]
    scale: str
    def __init__(self, data: _Optional[_Iterable[_Union[CompatibleAnime, _Mapping]]] = ..., scale: _Optional[str] = ...) -> None: ...

class CompatibilityRequest(_message.Message):
    __slots__ = ("target_anime_id", "user_favourite_ids")
    TARGET_ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    USER_FAVOURITE_IDS_FIELD_NUMBER: _ClassVar[int]
    target_anime_id: int
    user_favourite_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, target_anime_id: _Optional[int] = ..., user_favourite_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class CompatibilityResponse(_message.Message):
    __slots__ = ("target_anime_id", "compatibility_score", "scale")
    TARGET_ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    target_anime_id: int
    compatibility_score: float
    scale: str
    def __init__(self, target_anime_id: _Optional[int] = ..., compatibility_score: _Optional[float] = ..., scale: _Optional[str] = ...) -> None: ...

class CompatibilityBatchRequest(_message.Message):
    __slots__ = ("target_anime_ids", "user_favourite_ids")
    TARGET_ANIME_IDS_FIELD_NUMBER: _ClassVar[int]
    USER_FAVOURITE_IDS_FIELD_NUMBER: _ClassVar[int]
    target_anime_ids: _containers.RepeatedScalarFieldContainer[int]
    user_favourite_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, target_anime_ids: _Optional[_Iterable[int]] = ..., user_favourite_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class CompatibilityBatchResponse(_message.Message):
    __slots__ = ("scores",)
    SCORES_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.RepeatedCompositeFieldContainer[CompatibilityResponse]
    def __init__(self, scores: _Optional[_Iterable[_Union[CompatibilityResponse, _Mapping]]] = ...) -> None: ...

class CollaborativeRecommendationRequest(_message.Message):
    __slots__ = ("user_favourite_ids", "limit")
    USER_FAVOURITE_IDS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    user_favourite_ids: _containers.RepeatedScalarFieldContainer[int]
    limit: int
    def __init__(self, user_favourite_ids: _Optional[_Iterable[int]] = ..., limit: _Optional[int] = ...) -> None: ...

class CollaborativeRecommendationResponse(_message.Message):
    __slots__ = ("recommended",)
    RECOMMENDED_FIELD_NUMBER: _ClassVar[int]
    recommended: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, recommended: _Optional[_Iterable[int]] = ...) -> None: ...

class RelatedRequest(_message.Message):
    __slots__ = ("anime_id", "limit")
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    anime_id: int
    limit: int
    def __init__(self, anime_id: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...

class RelatedResponse(_message.Message):
    __slots__ = ("anime_ids",)
    ANIME_IDS_FIELD_NUMBER: _ClassVar[int]
    anime_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, anime_ids: _Optional[_Iterable[int]] = ...) -> None: ...
