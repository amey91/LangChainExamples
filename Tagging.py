from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

class Tagging(BaseModel):
	"""Tag the piece of text with particular info."""
	sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
	language: str = Field(description="language of text (should be ISO 639-1 code)")

convert_pydantic_to_openai_function(Tagging)

model = ChatOpenAI(temperature=0) # for deterministic output
tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages([
	("system", "Think carefully, and then tag the text as instructed"),
	("user", "{input}")
])

model_with_functions = model.bind(
	functions=tagging_functions,
	function_call={"name": "Tagging"} # forces to ALWAYS do the tagging
)

tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

tagging_chain.invoke({"input": "I love langchain"})

