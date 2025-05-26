import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)



#2 messages = [
	{
    	"role": "user",
    	"content": "What's the weather like in Boston?"
	}
]

# 3 Call the ChatCompletion endpoint
response = openai.ChatCompletion.create(
	# OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
	model="gpt-3.5-turbo",
	messages=messages,
	functions=functions  # OR use model.bind(functions) to do it just once and not pass in every time.
)

# 4 call the function on your side
args = json.loads(response[“choices”][0][“message”][“function_call”][“args”])
observation = get_current_weather(args)

# 5 append result to chat history and call OpenAI again
messages.append(
	{
		role = function,
		name = get_current_weather,
		content = observation
}
)

# 6 call openAI again to get the observation formatted correctly
response = OpenAI.ChatCompletion.create(
messages=messages,
model = someModel,
)	

#7 response will be something like response[choises][0][message][content]
E.g. “The current weather in Boston is 75* and its windy”

