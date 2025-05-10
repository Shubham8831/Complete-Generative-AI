from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books', # path of folder
    glob='*.pdf', # pattern of what files to select
    loader_cls=PyPDFLoader  # what loader class to use for loading the files
)

# docs = loader.load()  
# print(docs[0].page_content())


docs = loader.lazy_load()


for document in docs:
    print(document.metadata)