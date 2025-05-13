from langchain_community.retrievers import WikipediaRetriever
retriver = WikipediaRetriever(top_k_results= 2, lang= "en") # top_k_results - how much document u want in return

#define your query
query = "artificial intelligence"

#get relevant wikipedia content
docs = retriver.invoke(query)

print(docs[0].page_content) 