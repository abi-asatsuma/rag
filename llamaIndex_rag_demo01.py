import os
import huggingface_hub as hf_hub
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.prompts import PromptTemplate
import chromadb

# --- モデルのダウンロード ---
llm_model_id = "OpenVINO/qwen2.5-1.5b-instruct-int4-ov"
llm_path = "./qwen2.5_ov_model"
if not os.path.exists(llm_path):
    hf_hub.snapshot_download(llm_model_id, local_dir=llm_path)

# --- モデルの初期化 ---

embed_model = OpenVINOEmbedding(
    model_id_or_path="intfloat/multilingual-e5-small",
    device="gpu",
)

# ChromaDBセットアップ
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#--- ドキュメントの読み込みとインデックス作成 ---
documents = SimpleDirectoryReader("docs").load_data()

if chroma_collection.count() > 0: # 既存データがある場合は再利用
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model,
    )
else:
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model,
        overwrite=True  # 既存データがない場合は新規作成
    )

# LLM: ストップトークンとパラメータの調整
Settings.llm = OpenVINOLLM(
    model_id_or_path=llm_path,
    device_map="GPU",
    context_window=4096,
    max_new_tokens=256, # 簡潔にするために短く設定
    generate_kwargs={
        "do_sample": False,
        "repetition_penalty": 1.1, # 同じことを繰り返さないように
    }
)

# --- プロンプトの日本語化・簡潔化 ---
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in Japanese as concisely as possible.\n"
    "Query: {query_str}\n"
    "Answer (Japanese): "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# --- RAGの実行 ---
query_engine = index.as_query_engine(
    similarity_top_k=2,
)
# プロンプトを更新
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

response = query_engine.query("会社の飲み会のルールついてドキュメントを参照して日本語で簡潔に教えて")

print("\n--- Answer ---")
print(response)
