CREATE DATABASE IF NOT EXISTS assistantdb;

CREATE USER IF NOT EXISTS assistantRAG IDENTIFIED WITH plaintext_password BY 'safety_password';

GRANT ALL ON assistantdb.* TO assistantRAG;

USE assistantdb;

CREATE TABLE IF NOT EXISTS rag
(
    id      UUID,

    class_1 String,
    class_2 String,
    text    String,
    text_embeddings Array(Float32),

    urls Array(String),

    image_path String,
    image_embeddings Array(Float32)
) ENGINE MergeTree() ORDER BY id;