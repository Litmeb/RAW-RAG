"""pgvector-backed embedding storage for the local RAG adapter."""

import asyncio
import hashlib
import re
import threading
from typing import Any, Dict, Iterable, List, Optional


class RAGPgVectorStore:
    """Small synchronous psycopg wrapper exposed through async methods."""

    _IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        database_url: str,
        table_name: str,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        if not database_url:
            raise ValueError(
                "RAG pgvector storage is enabled, but storage.pgvector.database_url "
                "or RAG_PGVECTOR_DATABASE_URL is empty."
            )
        self._validate_identifier(table_name, "storage.pgvector.table_name")
        self.database_url = database_url
        self.table_name = table_name
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._conn = None
        self._lock = threading.Lock()

    @classmethod
    def _validate_identifier(cls, value: str, label: str) -> None:
        if not cls._IDENTIFIER_RE.match(value):
            raise ValueError(f"{label} must be a simple SQL identifier, got {value!r}.")

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    async def upsert_conversation(
        self,
        conversation_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        await asyncio.to_thread(
            self._upsert_conversation_sync,
            conversation_id,
            chunks,
            embeddings,
        )

    async def search(
        self,
        conversation_id: str,
        query_embedding: List[float],
        top_k: int,
        allowed_chunk_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_sync,
            conversation_id,
            query_embedding,
            top_k,
            allowed_chunk_ids,
        )

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "storage_backend": "pgvector",
            "pgvector_collection": self.collection_name,
            "pgvector_table": self.table_name,
        }

    def _connect(self):
        if self._conn is None or self._conn.closed:
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError(
                    "RAG pgvector storage requires psycopg. Install the EverCore "
                    "dependencies, including psycopg[binary,pool]."
                ) from exc
            self._conn = psycopg.connect(self.database_url)
            self._conn.autocommit = True
        return self._conn

    def _initialize_sync(self) -> None:
        from psycopg import sql

        with self._lock:
            conn = self._connect()
            table = sql.Identifier(self.table_name)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {table} (
                            collection TEXT NOT NULL,
                            conversation_id TEXT NOT NULL,
                            chunk_id TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            embedding_model TEXT NOT NULL,
                            content_hash TEXT NOT NULL,
                            embedding vector NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            PRIMARY KEY (collection, conversation_id, chunk_id)
                        )
                        """
                    ).format(table=table)
                )

    def _upsert_conversation_sync(
        self,
        conversation_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        from psycopg import sql

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Cannot store pgvector embeddings for {conversation_id}: "
                f"{len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
        rows = [
            (
                self.collection_name,
                conversation_id,
                str(chunk["chunk_id"]),
                int(chunk.get("chunk_index", idx)),
                self.embedding_model,
                self._content_hash(chunk.get("content", "")),
                self._vector_literal(embedding),
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        with self._lock:
            conn = self._connect()
            table = sql.Identifier(self.table_name)
            if embeddings:
                self._ensure_index_sync(len(embeddings[0]))
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        DELETE FROM {table}
                        WHERE collection = %s
                          AND conversation_id = %s
                          AND NOT (chunk_id = ANY(%s))
                        """
                    ).format(table=table),
                    (self.collection_name, conversation_id, chunk_ids),
                )
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {table} (
                            collection,
                            conversation_id,
                            chunk_id,
                            chunk_index,
                            embedding_model,
                            content_hash,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (collection, conversation_id, chunk_id)
                        DO UPDATE SET
                            chunk_index = EXCLUDED.chunk_index,
                            embedding_model = EXCLUDED.embedding_model,
                            content_hash = EXCLUDED.content_hash,
                            embedding = EXCLUDED.embedding,
                            updated_at = now()
                        WHERE {table}.chunk_index IS DISTINCT FROM EXCLUDED.chunk_index
                           OR {table}.embedding_model IS DISTINCT FROM EXCLUDED.embedding_model
                           OR {table}.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                           OR {table}.embedding::text IS DISTINCT FROM EXCLUDED.embedding::text
                        """
                    ).format(table=table),
                    rows,
                )

    def _search_sync(
        self,
        conversation_id: str,
        query_embedding: List[float],
        top_k: int,
        allowed_chunk_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        from psycopg import sql

        vector_literal = self._vector_literal(query_embedding)
        dimension = len(query_embedding)
        with self._lock:
            conn = self._connect()
            table = sql.Identifier(self.table_name)
            self._ensure_index_sync(dimension)
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        WITH visible AS MATERIALIZED (
                            SELECT chunk_id, chunk_index, embedding
                            FROM {table}
                            WHERE collection = %s
                              AND conversation_id = %s
                              AND embedding_model = %s
                              AND (%s::text[] IS NULL OR chunk_id = ANY(%s::text[]))
                        )
                        SELECT
                            chunk_id,
                            chunk_index,
                            1 - (embedding::vector({dimension}) <=> %s::vector({dimension})) AS similarity_score
                        FROM visible
                        ORDER BY embedding::vector({dimension}) <=> %s::vector({dimension})
                        LIMIT %s
                        """
                    ).format(table=table, dimension=sql.SQL(str(dimension))),
                    (
                        self.collection_name,
                        conversation_id,
                        self.embedding_model,
                        allowed_chunk_ids,
                        allowed_chunk_ids,
                        vector_literal,
                        vector_literal,
                        top_k,
                    ),
                )
                return [
                    {
                        "chunk_id": row[0],
                        "chunk_index": row[1],
                        "similarity_score": float(row[2]),
                    }
                    for row in cur.fetchall()
                ]

    def _ensure_index_sync(self, dimension: int) -> None:
        from psycopg import sql

        conn = self._connect()
        table = sql.Identifier(self.table_name)
        index = sql.Identifier(self._index_name(dimension))
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index}
                    ON {table}
                    USING hnsw ((embedding::vector({dimension})) vector_cosine_ops)
                    WHERE embedding_model = {embedding_model}
                    """
                ).format(
                    index=index,
                    table=table,
                    dimension=sql.SQL(str(dimension)),
                    embedding_model=sql.Literal(self.embedding_model),
                )
            )

    def _index_name(self, dimension: int) -> str:
        model_hash = hashlib.sha1(self.embedding_model.encode("utf-8")).hexdigest()[:10]
        return f"{self.table_name}_hnsw_{model_hash}_{dimension}_idx"

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _vector_literal(values: Iterable[float]) -> str:
        return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"
