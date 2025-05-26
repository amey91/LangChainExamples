from langchain.schema.runnable import RunnablePassthrough

chain = prompt | model | OpenAIFunctionsAgentOutputParser()
agent_chain = RunnablePassthrough.assign(
	# adds agent_scratchpad to the input dict
	agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"]) 
) | chain

	from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent_chain, tools=tools)
agent_executor.invoke({"input": "what is langchain?"})


# Message history and maintaining context
prompt = ChatPromptTemplate.from_messages([
	("system", "You are helpful but sassy assistant"),
	# History comes before current user input
	MessagesPlaceholder(variable_name="chat_history"),
	("user", "{input}"),
	MessagesPlaceholder(variable_name="agent_scratchpad")
])

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)

agent_executor.invoke({"input": "my name is bob"})
