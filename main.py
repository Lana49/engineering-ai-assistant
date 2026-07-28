import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5.4-mini",
    input="write a haiku about ai",
    store=True,
)

print(response.output_text)
