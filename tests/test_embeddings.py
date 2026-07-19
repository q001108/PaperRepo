from src.embeddings import HashEmbeddingFunction, SentenceTransformerEmbeddingFunction


class FakeSentenceTransformerModel:
    def encode(self, documents, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return FakeEmbeddingArray([[1.0, 0.0], [0.0, 1.0]][: len(documents)])


class FakeEmbeddingArray:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_hash_embedding_returns_configured_dimension():
    embedding_fn = HashEmbeddingFunction(dimensions=8)

    embedding = embedding_fn(["hello world"])[0]

    assert len(embedding) == 8


def test_sentence_transformer_embedding_uses_model_encode():
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_fn._model = FakeSentenceTransformerModel()

    embeddings = embedding_fn(["paper", "repo"])

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
