from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

from langchain_community.document_loaders import TextLoader # to load text form specific .txt file hehehe :)

load_dotenv()

# # loading text
# # Safer way to open a file

loader  = TextLoader("C:\\Users\\shubu\\Desktop\\GenAI\\7.Chains\\parallel_quiz_document.txt", encoding="utf-8")
text = loader.load()

template1 = PromptTemplate(
    template = "generate short, simple and easy notes on the following text. \n {text}",
    input_variables=["text"]
)

template2 = PromptTemplate(
    template = "from the following text : {text}, generate 10 quiz questions and answer.",
    input_variables = ["text"]
)
template3 = PromptTemplate(
    template="""
You are a helpful assistant. Below are study notes and quiz questions generated from a source text.

Your task is to format them into a single structured document with clear headers.

Notes:
{notes}

Quiz:
{quiz}

Return the final merged document in this format:

---
### Notes : notes here


---
### Quiz : quiz here

""",
    input_variables=["notes", "quiz"]
)


model1 = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)

model2 = OllamaLLM(
    model = "gemma3:1b"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': template1 | model1 | parser,
    'quiz': template2 | model2 | parser
})

merge_chain = template3 | model1 | parser

chain = parallel_chain | merge_chain



# or we can try a simple way to enter text here:


# text = '''
# Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

# The advantages of support vector machines are:

# Effective in high dimensional spaces.

# Still effective in cases where number of dimensions is greater than the number of samples.

# Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

# Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

# The disadvantages of support vector machines include:

# If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

# SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

# The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
# '''



result = chain.invoke({'text':text})

print(result)