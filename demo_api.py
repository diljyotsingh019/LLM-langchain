from langchain_openai import ChatOpenAI
import os
import asyncio
from typing import Any
import uvicorn

from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from warnings import filterwarnings

filterwarnings("ignore")
os.environ["OPENAI_API_KEY"] = ""


app = FastAPI()

llm = ChatOpenAI(
    temperature = 0,
    model_name = "gpt-3.5-turbo",
    streaming = True,
    callbacks = []
)

memory = ConversationBufferWindowMemory(memory_key= "chat_history",
                                        k = 5,
                                        model_name = "gpt-3.5-turbo",
                                        return_messages = True,
                                        output_key= "output")

agent = initialize_agent(
    agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory = memory,
    llm=llm,
    verbose = True,
    early_stopping_methpd = "generate",
    max_iterations = 5,
    tools = [],
    return_intermediate_steps = False
)

async def run_call(query: str, stream_it : AsyncIteratorCallbackHandler):
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    response = await agent.acall(inputs = {"input": query})
    return response

async def create_gen(query:str, stream_it: AsyncIteratorCallbackHandler):
    task = asyncio.create_task(run_call(query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task

class Query(BaseModel):
    text: str

@app.get("/chat")
async def chat(query: Query = Body(...)):
    stream_it = AsyncIteratorCallbackHandler()
    gen = create_gen(query.text, stream_it)
    return StreamingResponse(gen, media_type= "text/event-stream")

@app.get("/health")
async def health():
    return {"Status": "OK"}


if __name__ == "__main__":
    uvicorn.run("app:app", host = "localhost", port = 8000, reload = True)