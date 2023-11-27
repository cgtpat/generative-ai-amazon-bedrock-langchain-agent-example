from langchain.agents.tools import Tool
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents import AgentExecutor
from tools import tools
from datetime import datetime

PREFIX = "\n\nHuman: You are a Q&A AI chatbot (Assistant) for a company called Kuehne + Nagel eShipAsia. Also, you can answer general questions about eShipAsia and other information from your context. \
If you do not have any context or information on something, you inform the client about it politely.\
Assistant:"

FORMAT_INSTRUCTIONS = "\n\nHuman: \n\nAssistant:"

class FSIAgent():
    def __init__(self,llm, memory) -> None:
        self.prefix = PREFIX
        self.ai_prefix = "Assistant"
        self.human_prefix = "Human"
        self.llm = llm
        self.memory = memory
        self.format_instructions = FORMAT_INSTRUCTIONS
        self.agent = self.create_agent()

    def create_agent(self):
        fsi_agent = ConversationalAgent.from_llm_and_tools(
            llm = self.llm,
            tools = tools,
            prefix = self.prefix,
            ai_prefix = self.ai_prefix,
            human_prefix = self.human_prefix,
            format_instructions = self.format_instructions,
            return_intermediate_steps = True,
            return_source_documents = True
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=fsi_agent, tools=tools, verbose=True, memory=self.memory, return_source_documents=True, return_intermediate_steps=True) # , handle_parsing_errors=True
        return agent_executor

    def run(self, input):
        print("Running FSI Agent with input: " + str(input))
        try:
            response = self.agent(input)
        except ValueError as e:
            response = str(e)
            
            if not response.startswith("An output parsing error occurred"):
                raise e

            response = response.removeprefix("An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: Could not parse LLM output: `").removesuffix("`")
        
        return response
