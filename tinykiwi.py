import readline # activates input line editing (using arrow keys)
import faiss  # if your IDE is panicing about a missing faiss argument, ignore it. (check this -> https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
import numpy as np
from openai import OpenAI


d = 256 # size of a the embedding matrix - depends on what model you're using
embedding_index_db = faiss.IndexFlatL2(d) # build the index, d=size of vectors
embedding_text_db = []
embedding_index_db_search_return_size = 7

client = OpenAI(api_key="<your-api-key>") # this is just for local testing. Use .env instead

welcome_msg = "TinyKiwi: Hello There, I'm Tinykiwi!"
print(welcome_msg)
user_i = welcome_msg +"\n"+ input("You: ")

# text embedding _________________________________
def embedding_generate(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text, encoding_format="float") # encoding_format="float" gives reduced embedding size
    return np.array([response.data[0].embedding[:256]], dtype=np.float32) # faiss expects a float32 np matrix of size n-by-d, in this case n=1, d=256


def embedding_index_db_add(text):
    embedding_text_db.append(text)
    embedding_index_db.add(embedding_generate(text))


def embedding_index_db_search_similar(embd):
    _, indexes = embedding_index_db.search(embd, embedding_index_db_search_return_size)
    indexes = indexes.flatten() # [[i]] -> [i]
    #print(indexes) # --- for debugging
    return [(embedding_text_db[i] if i > 0 else "") for i in indexes] # faiss returns -1 if no similar embedding was found


while(user_i != "q"):
    new_embedding = embedding_generate(user_i)
    query = "acting as a chatbot, here're some pervious interactions with the user to improve your responses:\n" + \
            "\n".join(embedding_index_db_search_similar(new_embedding)) + \
            "\n Using the previous messages as a reference, respond to this message: \n" + user_i
    # print(f"\n\n \x1b[38;5;83m{query}\x1b[0m \n\n") # --- for debugging
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "You're Tinykiwi, a chat bot that enjoys chatting about random stuff."
        },
        {
        "role": "user",
        "content": query
        }
    ],
    temperature=0.7,
    max_tokens=64,
    top_p=1)
    result = response.choices[0].message.content
    interaction = F"user: {user_i}\nchatbot: {result}"
    embedding_index_db_add(interaction)
    print(F"Tinykiwi: {result}")
    user_i = input("You: ")
