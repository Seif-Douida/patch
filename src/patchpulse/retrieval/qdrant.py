"""The review collection on Qdrant (design D2): its settings, its alias, BM25 options per Steam
language, and exact counts per day for the nightly reconciliation (design D5).

Qdrant computes the BM25 vectors itself from each point's text (model `qdrant/bm25`); the options
sent with the text pick the tokenizer, stemmer and stopwords for the review's language. A query is
sent with its own language's options, so matching across languages is the dense vector's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr
from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "bm25"
BM25_MODEL = "qdrant/bm25"
# BM25's expected document length, in terms: an indexed review keeps 41 distinct terms on average
# once stopwords go (2,000 eligible reviews of the 12 Oct snapshot), so about 50 with repeats.
BM25_AVG_LEN = 50.0
TIMEOUT_S = 30
# More days than a game can have in the index, so a facet is never cut short.
FACET_LIMIT = 100_000

# Indexed payload fields (design D2); the text, version and fingerprint are stored, not indexed.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "appid": models.PayloadSchemaType.INTEGER,
    "day": models.PayloadSchemaType.INTEGER,
    "created_ts": models.PayloadSchemaType.INTEGER,
    "language": models.PayloadSchemaType.KEYWORD,
    "voted_up": models.PayloadSchemaType.BOOL,
    "playtime_bucket": models.PayloadSchemaType.KEYWORD,
    "aspects": models.PayloadSchemaType.KEYWORD,
    "aspect_model_version": models.PayloadSchemaType.KEYWORD,
    "first_seen_run_id": models.PayloadSchemaType.INTEGER,
}

_WORD = models.TokenizerType.WORD
# Chinese, Japanese and Thai don't put spaces between words; Korean does, but its words carry
# particles. Qdrant's multilingual tokenizer segments all four.
_MULTI = models.TokenizerType.MULTILINGUAL
_S = models.SnowballLanguage
_L = models.Language

# Steam language code -> (tokenizer, Snowball stemmer, stopword list); None where Qdrant has none.
# Every code Steam's review API uses, plus `malay`, which the snapshot holds too.
STEAM_BM25: dict[str, tuple[models.TokenizerType, models.SnowballLanguage | None, _L | None]] = {
    "arabic": (_WORD, _S.ARABIC, _L.ARABIC),
    "brazilian": (_WORD, _S.PORTUGUESE, _L.PORTUGUESE),
    "bulgarian": (_WORD, None, None),
    "czech": (_WORD, None, None),
    "danish": (_WORD, _S.DANISH, _L.DANISH),
    "dutch": (_WORD, _S.DUTCH, _L.DUTCH),
    "english": (_WORD, _S.ENGLISH, _L.ENGLISH),
    "finnish": (_WORD, _S.FINNISH, _L.FINNISH),
    "french": (_WORD, _S.FRENCH, _L.FRENCH),
    "german": (_WORD, _S.GERMAN, _L.GERMAN),
    "greek": (_WORD, _S.GREEK, _L.GREEK),
    "hungarian": (_WORD, _S.HUNGARIAN, _L.HUNGARIAN),
    "indonesian": (_WORD, None, _L.INDONESIAN),
    "italian": (_WORD, _S.ITALIAN, _L.ITALIAN),
    "japanese": (_MULTI, None, _L.JAPANESE),
    "koreana": (_MULTI, None, None),
    "latam": (_WORD, _S.SPANISH, _L.SPANISH),
    "malay": (_WORD, None, None),
    "norwegian": (_WORD, _S.NORWEGIAN, _L.NORWEGIAN),
    "polish": (_WORD, None, None),
    "portuguese": (_WORD, _S.PORTUGUESE, _L.PORTUGUESE),
    "romanian": (_WORD, _S.ROMANIAN, _L.ROMANIAN),
    "russian": (_WORD, _S.RUSSIAN, _L.RUSSIAN),
    "schinese": (_MULTI, None, _L.CHINESE),
    "spanish": (_WORD, _S.SPANISH, _L.SPANISH),
    "swedish": (_WORD, _S.SWEDISH, _L.SWEDISH),
    "tchinese": (_MULTI, None, _L.CHINESE),
    "thai": (_MULTI, None, None),
    "turkish": (_WORD, _S.TURKISH, _L.TURKISH),
    "ukrainian": (_WORD, None, None),
    "vietnamese": (_WORD, None, None),
}


class CollectionMismatchError(RuntimeError):
    """An existing collection's settings differ from the design; it's never altered in place."""


@dataclass(frozen=True)
class CollectionSpec:
    """The collection's settings (design D2), as `create_collection` takes them."""

    vectors: dict[str, models.VectorParams]
    sparse_vectors: dict[str, models.SparseVectorParams]
    quantization: models.ScalarQuantization
    hnsw: models.HnswConfigDiff
    on_disk_payload: bool


def make_client(url: str, api_key: SecretStr | str | None, *, grpc: bool = False) -> QdrantClient:
    """A REST client with a 30 s timeout, or with `grpc` one that sends what it can over gRPC
    (port 6334): the laptop's upload, where a vector is 1.5 KB of float32 instead of 8.4 KB of
    JSON. The key stays inside the client: never in its repr or in the errors it raises."""
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    # No version check on creation: it's an extra request, and Task 17 matches versions instead.
    return QdrantClient(
        url=url, api_key=key, timeout=TIMEOUT_S, prefer_grpc=grpc, check_compatibility=False
    )


def collection_config(dense_dim: int) -> CollectionSpec:
    """Full dense vectors on disk with an int8 copy in RAM, BM25 with IDF, payload on disk.

    The BM25 index and the HNSW graph are on disk too, read through the page cache: design D9's
    fallback for the free cluster's 1 GB, taken during the upload (plan Task 17) when Qdrant's
    memory at a third of the points projected past 70% at rest and near the limit while indexing.
    """
    return CollectionSpec(
        vectors={
            DENSE: models.VectorParams(
                size=dense_dim, distance=models.Distance.COSINE, on_disk=True
            )
        },
        sparse_vectors={
            SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF, index=models.SparseIndexParams(on_disk=True)
            )
        },
        quantization=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
        ),
        hnsw=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=True),
        on_disk_payload=True,
    )


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """Create the collection with its payload indexes, or check that an existing one matches the
    design. A mismatch raises and names each difference."""
    spec = collection_config(dense_dim)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=spec.vectors,
            sparse_vectors_config=spec.sparse_vectors,
            quantization_config=spec.quantization,
            hnsw_config=spec.hnsw,
            on_disk_payload=spec.on_disk_payload,
        )
        for field, schema in PAYLOAD_INDEXES.items():
            client.create_payload_index(name, field, schema, wait=True)
        return
    differences = _differences(client.get_collection(name), spec)
    if differences:
        raise CollectionMismatchError(
            f"collection {name} differs from the design: {'; '.join(differences)}"
        )


def _differences(info: models.CollectionInfo, spec: CollectionSpec) -> list[str]:
    found: list[str] = []

    def compare(what: str, actual: object, expected: object) -> None:
        if actual != expected:
            found.append(f"{what} {actual} (design: {expected})")

    params = info.config.params
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    dense, want = vectors.get(DENSE), spec.vectors[DENSE]
    if dense is None:
        found.append(f"no {DENSE} vector")
    else:
        compare(f"{DENSE}: size", dense.size, want.size)
        compare(f"{DENSE}: distance", dense.distance, want.distance)
        compare(f"{DENSE}: on_disk", dense.on_disk, want.on_disk)
    sparse = (params.sparse_vectors or {}).get(SPARSE)
    compare(f"{SPARSE}: modifier", sparse and sparse.modifier, spec.sparse_vectors[SPARSE].modifier)
    compare(f"{SPARSE}: index on_disk", sparse and sparse.index and sparse.index.on_disk, True)
    quantization = info.config.quantization_config
    scalar = quantization.scalar if isinstance(quantization, models.ScalarQuantization) else None
    compare(
        "quantization", scalar and (scalar.type, scalar.always_ram), (models.ScalarType.INT8, True)
    )
    compare("on_disk_payload", params.on_disk_payload, spec.on_disk_payload)
    hnsw = info.config.hnsw_config
    compare(
        "hnsw",
        (hnsw.m, hnsw.ef_construct, bool(hnsw.on_disk)),
        (spec.hnsw.m, spec.hnsw.ef_construct, spec.hnsw.on_disk),
    )
    schema = {field: index.data_type for field, index in info.payload_schema.items()}
    for field, expected in PAYLOAD_INDEXES.items():
        compare(f"payload index {field}:", schema.get(field), expected)
    for field in sorted(set(schema) - set(PAYLOAD_INDEXES)):
        found.append(f"payload index {field}: not in the design")
    return found


def point_alias(client: QdrantClient, alias: str, collection: str) -> None:
    """Point `alias` at `collection` in one request, so readers never see it missing."""
    operations: list[models.AliasOperations] = []
    if any(a.alias_name == alias for a in client.get_aliases().aliases):
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)


def bm25_options(language: str) -> models.Bm25Config:
    """BM25 options for a Steam language code, every one spelled out: Qdrant assumes English for
    anything left unset. A code without an entry gets plain word splitting, no stemmer and no
    stopwords."""
    tokenizer, stemmer, stopwords = STEAM_BM25.get(language, (_WORD, None, None))
    return models.Bm25Config(
        avg_len=BM25_AVG_LEN,
        tokenizer=tokenizer,
        stemmer=(
            models.SnowballParams(type=models.Snowball.SNOWBALL, language=stemmer)
            if stemmer is not None
            else models.DisabledStemmerParams(type=models.NoStemmer.NONE)
        ),
        stopwords=stopwords
        if stopwords is not None
        else models.StopwordsSet(languages=[], custom=[]),
    )


def facet_counts(client: QdrantClient, collection: str, appid: int) -> dict[int, int]:
    """Exact point counts per `day` for one game."""
    response = client.facet(
        collection,
        key="day",
        facet_filter=models.Filter(
            must=[models.FieldCondition(key="appid", match=models.MatchValue(value=appid))]
        ),
        exact=True,
        limit=FACET_LIMIT,
    )
    if len(response.hits) >= FACET_LIMIT:
        raise RuntimeError(f"facet on day for {appid} reached its limit of {FACET_LIMIT}")
    return {int(hit.value): hit.count for hit in response.hits}
