from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

model_id = 'mistralai/mixtral-8x7b-instruct-v01'

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = "skills-network"

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

mixtral_llm = WatsonxLLM(model=model)
mixtral_llm

template = """Tell me a {adjective} joke about {content}.
"""
prompt = PromptTemplate.from_template(template)

prompt.format(adjective="funny", content="chickens")

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm)
response = llm_chain.invoke(input = {"adjective": "funny", "content": "chickens"})
print(response["text"])

response = llm_chain.invoke(input = {"adjective": "sad", "content": "fish"})
print(response["text"])


# Summarization
content = """
        The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation. 
        Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems. 
        For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion. 
        Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers. 
        These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
"""

template = """Summarize the {content} in one sentence.
"""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm)
response = llm_chain.invoke(input = {"content": content})
print(response["text"])


# Question and Answer

content = """
        The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
        The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
        The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
"""

question = "Which planets in the solar system are rocky and solid?"

template = """
            Answer the {question} based on the {content}.
            Respond "Unsure about answer" if not sure about the answer.
            
            Answer:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "answer"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"question":question ,"content": content})
print(response["answer"])

# Text Classification
text = """
        The concert last night was an exhilarating experience with outstanding performances by all artists.
"""

categories = "Entertainment, Food and Dining, Technology, Literature, Music."

template = """
            Classify the {text} into one of the {categories}.
            
            Category:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "category"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"text":text ,"categories": categories})
print(response["category"])

# Code generation
description = """
        Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
        The table 'purchases' contains a column 'purchase_date'
"""

template = """
            Generate an SQL query based on the {description}
            
            SQL Query:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "query"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"description":description})
print(response["query"])

# Role Playing 
role = """
        game master
"""

tone = "engaging and immersive"

template = """
            You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
            
            Answer:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "answer"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"role": role, "question": "What can I do in the game?", "tone": tone})




