from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3


bedrock_client=boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

model_id= "meta.llama3-70b-instruct-v1:0"

llm = Bedrock(
    model_id=model_id,
    client=bedrock_client
)

def my_model(user_prompt):
    prompt=PromptTemplate(
        input_variables=['user_prompt'],
        template="you are a chatbot. provide ans for {user_prompt}"
    )


    bedrock_chain = LLMChain(llm=llm,prompt=prompt)
    response=bedrock_chain({'user_prompt':user_prompt})

    return response


user_prompt="what is python?"
res= my_model(user_prompt)

print(res['text'])

