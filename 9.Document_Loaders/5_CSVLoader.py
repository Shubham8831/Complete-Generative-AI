from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='C:\\Users\\shubu\\Desktop\\GenAI\\9.Document_Loaders\\csv_file.csv')

docs = loader.load()

print(len(docs))
print(docs[1])