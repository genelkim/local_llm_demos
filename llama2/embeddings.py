from langchain_community.embeddings import LlamaCppEmbeddings

# Make sure the model path is correct for your system!
llm = LlamaCppEmbeddings(
    model_path="/home/gene/research/llama-2/models/llama-2-13b.Q4_K_M.gguf",
    verbose=True,
    n_gpu_layers=41,
)

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

