from pydantic import BaseModel, Field
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain.utilities.openapi import OpenAPISpec

class SearchInput(BaseModel):
		query: str = Field(description="Thing to search for")

	@tool(args_schema=SearchInput)
def search(query: str) -> str:
	"""Search for the weather online."""
	current_temp = $(API_CALL)
	return "current temperature is {current_temp} *f"

text = """
{
"openapi": "3.0.0",
"Info….
	paths": {
	"/pets": {
  	"get": {....
		 "post": {
    	"summary": "Create a pet",
    	"operationId": "createPets",
    	"tags": [
      	"pets"
    	],
    	"responses": {....
…
…
}"""

spec = OpenAPISpec.from_text(text)

pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)

model = ChatOpenAI(temperature=0).bind(functions=pet_openai_functions)

model.invoke("what are three pets names") 	# calls list with limit = 3

model.invoke("tell me about pet with id 42")  # calls get for id=42
