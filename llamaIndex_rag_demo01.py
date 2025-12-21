from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# 1. ローカルファイルからドキュメント読み込み
documents = SimpleDirectoryReader("docs").load_data()  # docs/配下にテキストやPDFを置く

# 2. ChromaDBセットアップ
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. LlamaIndexでインデックス作成
embed_model = OllamaEmbedding(model_name="nomic-embed-text")  # 任意のembeddingモデル
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model,
)

# 4. クエリ→検索→LLM生成（Ollamaを利用）
llm = Ollama(model="llama3:8b")  # 例: "llama3", "phi", "mistral" など
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
)
response = query_engine.query("有休取得について日本語で簡潔に教えて")

print(response)
