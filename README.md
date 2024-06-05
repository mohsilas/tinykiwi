# Tinykiwi
A tiny chatbot with a temporary chat-session memory using text embedding.

---
## TLDR: how does it work:
The fancy term for how tinykiwi works is retrieval augmented generation, which means that it feeds on previously-stored data to improve its responses. Basically, the chatbot consists of a text-embedding api, a vector database, and the gpt3.5 api. Text-embedding means turning text into an array of floats, preserving the relationships between words (meaning, context, or whatever you call it). Thus making searching for similar text a lot better. Tinykiwi embeds chunks of the conversation (user input + tinykiwi response) and then stores them in a Faiss (Facebook AI Similarity Search) database. When the user enters a new input, that input is seperatly embedded (turned to a float32 array) and used to retrieve the relevant chunks of the conversation. Which in turn are fed to the Openai api to augment its response.
## Usage
1. Clone the repository:
```
git clone https://github.com/mohsilas/tinykiwi
```
2. Install dependencies:
```
pip install openai
pip install faiss-cpu
pip install numpy
 ```
3. [get an Openai api key](https://openai.com/index/openai-api/)
4. run it:
```
python3 tinykiwi.py
```
## Some usefull links:
1. [Preserving context/session by refeeding responses](https://community.openai.com/t/how-to-preserve-the-context-session-of-a-conversation-with-the-api/324986/1)
2. [Openai's embedding model](https://platform.openai.com/docs/guides/embeddings/use-cases)
3. [Faiss indexing and searching](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)

## screenshot:
<img width="994" alt="mem_test" src="https://github.com/mohsilas/tinykiwi/assets/171826971/a1f6f746-ed4d-4caa-97a1-0ad93fbe9056">
