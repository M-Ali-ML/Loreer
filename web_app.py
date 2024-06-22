from llama_cpp import Llama
import textwrap
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import gc
import transformers
import os
import re 
import streamlit as st

import random
import time

# RAG
@st.cache_data
def read_data(data_path):
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def load_embeddings(embeddings_path):
    embeddings = torch.load(embeddings_path, map_location='cpu')
    return embeddings

@st.cache_resource
def load_embeddings_model(embedding_model_path, device='cpu'):
    model = SentenceTransformer(embedding_model_path, trust_remote_code=True, device=device)
    return model

def cosine_sim(question, embeddings, embeddings_model):
    query_embeddings = embeddings_model.encode(question, batch_size=1, show_progress_bar=False)
    cos_sim_score = cos_sim(a=query_embeddings, b=embeddings)[0]
    topk = torch.topk(cos_sim_score, k=5)
    return topk


def retriver(topk, data, print=False, min_score=0):
    context = []
    for k,i in enumerate(topk[1]):
        if topk[0][k] >= min_score:
            context.append(data.iloc[int(i)]["text"])
        else:
            break
    if print:
        for chunk in context:
            print("\n\n\n")
            print(textwrap.fill(chunk, width=120))
    return context

@st.cache_resource
def load_llm(path, n_ctx,verbose):

    llm = Llama(
        model_path=path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        flash_attn=True,
        verbose=verbose,
        )
    return llm


def prompt_extract_from_context(llm, question, context, temperature,max_tokens):

    summary_from_context = []
    for passage in context:
        extraction_prompt = f"""
        You are an AI assistance based on to this question: {question} 
        extract related information and summarize them from the given passage, 
        If the passage doesn't have related information, return [none].
        The passage: {passage}.
        If the passage doesn't have related information, return [none].
        """
        output = llm(
        extraction_prompt,
        max_tokens=max_tokens, # set to None to generate up to the end of the context window
        stop=["Q:"], # Stop generating just before the model would generate a new question
        echo=False,
        temperature=temperature,
    ) # Generate a completion, can also call create_completion
        
        summary_from_context.append(output["choices"][0]["text"])
    return summary_from_context

def combine_text(list_of_text, drop_word):
    filtered_list_with_indices = [(i, item) for i, item in enumerate(list_of_text) if not re.search(rf"{drop_word}", item, re.IGNORECASE)]
    
    # Separate the indices and items
    kept_indices, filtered_list = zip(*filtered_list_with_indices) if filtered_list_with_indices else ([], [])
    
    # Join the filtered list into a single string with spaces
    result = " ".join(filtered_list)
    
    return result, kept_indices

def get_reference(topk, data, kept_indices, site):
    data_index = topk[1][[kept_indices]].tolist()
    ref = site + data.iloc[data_index]["title"] 
    return ref

def prompt_chat(llm, question, filtered_summary, temperature,max_tokens):
    prompt_chat = f"""
    You are loreer, a wise and old AI assistance that knows all the stories about League of legends and its lore.
    With the following passage: {filtered_summary} from League of Legend wiki, answer this question:{question} in a very articulate and sophisticated way.
    Also don't mention that you have the passage beforehand, Instead, you will describe Zed's story and abilities as if you were loreer.
    
    """ 
    output = llm(
    prompt_chat,
    max_tokens=max_tokens, # set to None to generate up to the end of the context window
    stop=["Q:"], # Stop generating just before the model would generate a new question
    echo=False,
    temperature=temperature,
    ) # Generate a completion, can also call create_completion

    return output["choices"][0]["text"]
def add_reference(chat_output, ref):
    if len(ref) >0:
        ref.drop_duplicates(inplace=True)
        links = ""
        chat_output = chat_output + "\n\nReferences:\n " 
        for index,link in enumerate(ref):
            links = links + str(index + 1) +". " + link.replace(" ", "_") + "\n"
        chat_output = chat_output + links
    return chat_output

# combines all functions
def loreer(question,
           llm_model_path,
           data_path,
           embeddings_path,
           embedding_model_path,
           n_ctx=4096,
           max_tokens=[512,None], # first for extraction summary, second for question answering
           print_context=False,
           temperature=[0.1, 0.7], # first for extraction summary, second for question answering
           verbose=False,
           site= "https://leagueoflegends.fandom.com/wiki/",
           min_score=0.3):
    
       data = read_data(data_path)
       embeddings = load_embeddings(embeddings_path)
       embeddings_model = load_embeddings_model(embedding_model_path)
       topk = cosine_sim(question, embeddings, embeddings_model)
       context = retriver(topk, data, print=print_context,min_score=min_score)
       llm = load_llm(llm_model_path,n_ctx,verbose=verbose)
       summary_from_context = prompt_extract_from_context(llm, question, context, temperature[0],max_tokens[0])
       filtered_summary, kept_indices = combine_text(summary_from_context, "none")
       ref = get_reference(topk, data, kept_indices, site)
       chat_output = prompt_chat(llm, question, filtered_summary, temperature[1],max_tokens[1])
       chat_output = add_reference(chat_output, ref)
       return chat_output


# Streamlit

st.markdown("""
# Chat with Loreer 🧙‍♂️
knows more about LoL than your silver friend 😉
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# create a stream generator
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.03)

# Accept user input
if prompt := st.chat_input("Hello summoner, how can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


    results = loreer(
        question=prompt,
        llm_model_path="./models/meta-llama-3-8b-instruct.Q4_K_M.gguf",
        data_path="./data/chunks_data.csv",
        embeddings_path="./data/embeddings_torch.pt",
        embedding_model_path="Alibaba-NLP/gte-base-en-v1.5",
        n_ctx=6144,
        max_tokens=[512,None], # first for extraction summary, second for question answering
        print_context=False,
        temperature=[0, 0.7], # first for extraction summary, second for question answering
        site = "https://leagueoflegends.fandom.com/wiki/",
        verbose=False,

    )
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(stream_data(results))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
