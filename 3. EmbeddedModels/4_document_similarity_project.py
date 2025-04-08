from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents) # finding document embedding (5 vector and each vector is in 300D space)
query_embedding = embedding.embed_query(query) # querry embedding( single vector )

scores = cosine_similarity([query_embedding], doc_embeddings)[0] # values should be 2d list [result will be 2d list]

# Get the index and score of the highest value in the 'scores' list
# 1. enumerate(scores) pairs each score with its index → [(0, score1), (1, score2), ...]
# 2. list(...) makes it a list if not already
# 3. sorted(..., key=lambda x: x[1]) sorts the list based on score (x[1] = score)
# 4. [-1] gets the last item in the sorted list → the highest score and its index
# 5. index, score = ... unpacks the tuple into 'index' and 'score' variables
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)



