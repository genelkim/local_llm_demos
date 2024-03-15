from langchain.embeddings import LlamaCppEmbeddings
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Make sure the model path is correct for your system!
llm = LlamaCppEmbeddings(
    model_path="/home/gene/research/llms/gguf_llms/models/mixtral-8x7b-v0.1.Q4_K_M.gguf",
    verbose=True,
    n_ctx=2048,
    f16_kv=True, 
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""

# Example texts.
texts = [
    "Hello, my name is Gene",
    "Goodbye, see you tomorrow",
    "This is a sentence"
]


# Run texts.
results = llm.embed_documents(texts)
individual_results = [
    llm.embed_query(text)
    for text
    in texts
]

print("List Result")
print(results)

print("Individual Results")
print(individual_results)

print("same?")
print(results == individual_results)

