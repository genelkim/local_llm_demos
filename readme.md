# Running LLMs locally

## Step 1: Acquire your models
Use the `gguf` quantized versions of Llama-2 models from [TheBloke](https://huggingface.co/TheBloke). There are several versions to choose from - `TheBloke` helpfully lists pros and cons of these models. The link to download the model directly is found by right clicking the download symbol next to the model file in the `Files and Versions` tab on the huggingface repo. You can use `wget` to download the file. Below is the command to download a 4-bit version of `llama-2-13b`.

```wget https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_M.gguf```

The models directory here is meant to be a shared spot for downloading models. The files are in the gitignore since we don't want to introduce the large model weight files into the repo.

## Step 2: Install dependencies
The dependencies are listed in `requirements.txt`. Make sure you are using the latest version of both libraries.
```pip install -r requirements.txt```

### Step 2b: Install llama-cpp-python with CUBLAS
Now we'll reinstall llama-cpp-python with CUBLAS on.
```CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir```

## Step 3: Replace the model path
The direct path to the `.gguf` file for your model is entered to the `model_path` parameter in `LlamaCpp` in `gguf.py` and `LlamaCppEmbeddings` in `embeddings.py`. Make sure to update that path to match your own file system. `llama.cpp` seems to require an absolute path.

## Step 4: Run
You can lift the `llm` declaration and put it in place of any existing `langchain` `llm` instance. To try out the demo, use the following:

```python gguf.py```

# Note
This method theoretically should also work for the `70b` models, but there seems to be a bug on the `llama.cpp` side. Feel free to open a pull request or an issue if you find a way to fix that.

# Attribution
This is based on [Rik's llama 2 demo](https://github.com/infiniterik/llama_demo).

