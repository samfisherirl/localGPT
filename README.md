# Fork of LocalGPT: One-Click-Installer without Conda 🌐
### Including a fix for llama-cpp-py 

fix https://github.com/zylon-ai/private-gpt/issues/1584#issuecomment-1938464222 

run one-click-installer.bat

Before relying on this for a fix, make sure you install Visual Studio Build Tools, select C++ Desktop Developement, including optional Windows 10 SDK 

```bat

      CMake Error at CMakeLists.txt:3 (project):
        No CMAKE_CXX_COMPILER could be found.



      -- Configuring incomplete, errors occurred!

      *** CMake configuration failed
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for llama-cpp-python
Failed to build llama-cpp-python
ERROR: Could not build wheels for llama-cpp-python, which is required to install pyproject.toml-based projects```

error 2

```bat

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
binary_path: D:\localGPT-main\venv\lib\site-packages\bitsandbytes\cuda_setup\libbitsandbytes_cuda116.dll
CUDA SETUP: Loading binary D:\localGPT-main\venv\lib\site-packages\bitsandbytes\cuda_setup\libbitsandbytes_cuda116.dll...
CUDA extension not installed.
CUDA extension not installed.
2024-05-10 17:51:52,645 - INFO - run_localGPT.py:244 - Running on: cuda
2024-05-10 17:51:52,645 - INFO - run_localGPT.py:245 - Display Source Documents set to: False
2024-05-10 17:51:52,645 - INFO - run_localGPT.py:246 - Use history set to: False
2024-05-10 17:51:53,854 - INFO - SentenceTransformer.py:66 - Load pretrained SentenceTransformer: hkunlp/instructor-large
load INSTRUCTOR_Transformer
max_seq_length  512
2024-05-10 17:51:55,635 - INFO - run_localGPT.py:132 - Loaded embeddings from hkunlp/instructor-large
Here is the prompt used: input_variables=['context', 'question'] output_parser=None partial_variables={} template='<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful assistant, you will use the provided context to answer user questions.\nRead the given context before answering questions and think step by step. If you can not answer a user question based on \nthe provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n            Context: {context}\n            User: {question}<|start_header_id|>assistant<|end_header_id|>' template_format='f-string' validate_template=True
2024-05-10 17:51:55,825 - INFO - run_localGPT.py:60 - Loading Model: TheBloke/Llama-2-7b-Chat-GGUF, on: cuda
2024-05-10 17:51:55,825 - INFO - run_localGPT.py:61 - This action can take a few minutes!
2024-05-10 17:51:55,825 - INFO - load_models.py:38 - Using Llamacpp for GGUF/GGML quantized models
D:\localGPT-main\venv\lib\site-packages\huggingface_hub\file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Traceback (most recent call last):
  File "D:\localGPT-main\venv\lib\site-packages\langchain\llms\llamacpp.py", line 149, in validate_environment
    from llama_cpp import Llama
ModuleNotFoundError: No module named 'llama_cpp'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\localGPT-main\run_localGPT.py", line 285, in <module>
    main()
  File "D:\localGPT-main\venv\lib\site-packages\click\core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
  File "D:\localGPT-main\venv\lib\site-packages\click\core.py", line 1078, in main
    rv = self.invoke(ctx)
  File "D:\localGPT-main\venv\lib\site-packages\click\core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "D:\localGPT-main\venv\lib\site-packages\click\core.py", line 783, in invoke
    return __callback(*args, **kwargs)
  File "D:\localGPT-main\run_localGPT.py", line 252, in main
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
  File "D:\localGPT-main\run_localGPT.py", line 142, in retrieval_qa_pipline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
  File "D:\localGPT-main\run_localGPT.py", line 65, in load_model
    llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
  File "D:\localGPT-main\load_models.py", line 56, in load_quantized_model_gguf_ggml
    return LlamaCpp(**kwargs)
  File "D:\localGPT-main\venv\lib\site-packages\langchain\load\serializable.py", line 74, in __init__
    super().__init__(**kwargs)
  File "pydantic\main.py", line 339, in pydantic.main.BaseModel.__init__
  File "pydantic\main.py", line 1100, in pydantic.main.validate_model
  File "D:\localGPT-main\venv\lib\site-packages\langchain\llms\llamacpp.py", line 153, in validate_environment
    raise ImportError(
ImportError: Could not import llama-cpp-python library. Please install the llama-cpp-python library to use this embedding model: pip install llama-cpp-python

(venv) D:\localGPT-main>git clone https://github.com/abetlen/llama-cpp-python
Cloning into 'llama-cpp-python'...
remote: Enumerating objects: 7334, done.
remote: Counting objects: 100% (1836/1836), done.
remote: Compressing objects: 100% (564/564), done.
remote: Total 7334 (delta 1502), reused 1434 (delta 1224), pack-reused 5498
Receiving objects: 100% (7334/7334), 1.77 MiB | 16.22 MiB/s, done.
Resolving deltas: 100% (4752/4752), done.
```
