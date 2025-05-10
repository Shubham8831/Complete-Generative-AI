from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('C:\\Users\\shubu\\Desktop\\GenAI\\9.Document_Loaders\\pdf_file.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata) 