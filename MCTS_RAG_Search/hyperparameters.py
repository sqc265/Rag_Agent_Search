candidate_hyperparameters = [
    # QueryDecomposer
    [{"model_name": "gemini-2.5-flash"}, 
     {"model_name": "gpt-5-nano"},
     {}],
    # Retriever
    [{
        "model_name": "gemini-2.5-flash",
        "top_n": 5
     }, 
     {
        "model_name": "gpt-5-nano",
        "top_n": 5
     },
     {
        "model_name": "gemini-2.5-flash",
        "top_n": 10
     }, 
     {
        "model_name": "gpt-5-nano",
        "top_n": 10
     },
     {}],
    # Reranker
    [{
        "model_name": "gemini-2.5-flash",
        "top_k": 3
     }, 
     {
        "model_name": "gpt-5-nano",
        "top_k": 3
     },
     {
        "model_name": "gemini-2.5-flash",
        "top_k": 5
     }, 
     {
        "model_name": "gpt-5-nano",
        "top_k": 5
     },
     {}],
    # Extractor
    [{"model_name": "gemini-2.5-flash"}, 
     {"model_name": "gpt-5-nano"},
     {}],
    # Generator
    [{"model_name": "gemini-2.5-flash"}, 
     {"model_name": "gpt-5-nano"},
     {}]
]