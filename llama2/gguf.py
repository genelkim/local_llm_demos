from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# make the templates
template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Verbose is required to pass to the callback manager
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/gene/research/llama-2/models/llama-2-13b.Q4_K_M.gguf",
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048,
    f16_kv=True, 
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""

# Run the prompt
llm.invoke(prompt)

# You can use this in any langchain prompt in place of your existing llm
