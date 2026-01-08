from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
import chromadb
import os
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# APIキー設定
gemini_key = os.getenv("GEMINI_API_KEY")


# 1. ローカルファイルからドキュメント読み込み＆チャンク分割
documents = SimpleDirectoryReader("docs").load_data()  # docs/配下にテキストやPDFを置く
parser = SimpleNodeParser.from_defaults(chunk_size=128, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)

# 2. ChromaDBセットアップ
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. LlamaIndexでインデックス作成
embed_model = OllamaEmbedding(model_name="nomic-embed-text", batch_size=2)  # 任意のembeddingモデル
index = VectorStoreIndex(
    nodes,
    vector_store=vector_store,
    embed_model=embed_model,
)

# 4. クエリ→検索→LLM生成
llm = Gemini(model="gemini-2.5-flash", temperature=0)  
# ローカルLLMを使う場合は以下を有効化
# ローカルLLMでも動くが、メモリ要求が多いため非推奨
# llm = Ollama(model_name="llama3.1:8b", temperature=0)
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3, # 検索上位3件を取得
)
response = query_engine.query("飲み会のルールについてドキュメントを参照して日本語で簡潔に教えて")

print(response)
