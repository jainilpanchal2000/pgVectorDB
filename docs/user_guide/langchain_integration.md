# LangChain Integration

pgVectorDB is designed to work seamlessly with the LangChain ecosystem. While it operates as a standalone vector database, it fully implements the required interfaces to be used as a LangChain `VectorStore` retriever.

## Using as a LangChain Retriever

If you are building a LangChain chain or agent, you can easily convert your `pgVectorDB` instance into a LangChain `Retriever`.

```python
from pgvectordb import pgVectorDB
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 1. Initialize your DB
db = pgVectorDB(
    collection_name="knowledge_base",
    embedding_model=my_embeddings,
    connection_string="postgresql+asyncpg://user:pass@localhost/db"
)
await db.initialize()

# 2. Convert to LangChain Retriever
# You can pass search arguments like 'k' or 'filter' here
retriever = db.as_retriever(search_kwargs={"k": 5})

# 3. Setup a basic RAG chain in LangChain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Invoke the chain
response = await chain.ainvoke("What are the system requirements?")
print(response)

```

## Advanced Retrievers

Because pgVectorDB is fully compatible, you can wrap it in LangChain's advanced retrievers.

### MultiQueryRetriever
Generates multiple versions of the user's query using an LLM to overcome wording variations, executing all of them against pgVectorDB.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

```

### Contextual Compression Retriever
While pgVectorDB has native reranking, you can also use LangChain's Contextual Compression pipeline to extract relevant snippets from the returned documents.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=db.as_retriever(search_kwargs={"k": 10})
)

```

## Adding LangChain Documents

When working with LangChain Document Loaders (like PDFLoader or WebBaseLoader), they return LangChain `Document` objects. You can insert these directly into pgVectorDB.

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load data
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 3. Add to pgVectorDB
# Extract texts and metadata from the LangChain Document objects
texts = [doc.page_content for doc in all_splits]
metadatas = [doc.metadata for doc in all_splits]

await db.add_texts(texts=texts, metadatas=metadatas)

```
