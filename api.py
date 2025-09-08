import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = 'sk-m4qTDFE0dB5HAMdiBeCdBc469aA44e1bB744D540F0025353'
os.environ['OPENAI_BASE_URL'] = 'https://aiproxy.lmzgc.cn:8080/v1'

client = OpenAI()
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o"
)

print("API测试结果:")
print(chat_completion.choices[0].message.content)

