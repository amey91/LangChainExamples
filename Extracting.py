from typing import Optional
class Person(BaseModel):
	"""Information about a person."""
	name: str = Field(description="person's name")
	age: Optional[int] = Field(description="person's age")

class Information(BaseModel):
	"""Information to extract."""
	people: List[Person] = Field(description="List of info about people")


extraction_functions = [convert_pydantic_to_openai_function(Information)]

extraction_model = model.bind(
functions=extraction_functions, 
function_call={"name": "Information"}, # forces the model to always call function
)

prompt = ChatPromptTemplate.from_messages([
	("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
	("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()

# OR just extract the people from people list directly using a different parser

extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
