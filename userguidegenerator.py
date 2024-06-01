import cv2
import base64
import os
import numpy as np
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from dotenv import load_dotenv
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "X"))

# Load the video
video = cv2.VideoCapture("octaaisignup.mp4")

# Extract frames and convert to base64
base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

# Select frames for the prompt (every 50th frame)
selected_frames = base64Frames[0::50]

# Using OctaAI 
llm = OctoAIEndpoint(
        model="meta-llama-3-8b-instruct",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        
    )

template="""You are an assistant for helping to write good user guides. Use the following step by step actions to write a helpful user guide.
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Craft the prompt and send a request to GPT
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to create a user guide for. Generate a title and give me detailed instructions for what you are seeing on the screen.",
            *map(lambda x: {"image": x, "resize": 768}, selected_frames),
        ],
    },
]

params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 300,
}

result = client.chat.completions.create(**params)
response_content = result.choices[0].message.content

# Extract title and instructions from the response
title, instructions = response_content.split('\n', 1)

run1 = chain.invoke("Write a user guide based upon the following statement and do not include the prompt" + instructions)
print(run1)

# Print the result in chunks to avoid truncation in the console
# chunk_size = 1000
# for i in range(0, len(instructions), chunk_size):
#    print(instructions[i:i + chunk_size])

# Create a PDF document
pdf_file = "output.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter
c.setFont("Helvetica", 14)

# Add the title to the PDF
c.drawString(40, height - 40, title)

# Add the first frame to the PDF
first_frame = base64Frames[0]
img_data = base64.b64decode(first_frame)
image = ImageReader(BytesIO(img_data))
c.drawImage(image, 40, height - 340, width=512, height=288)

# Add the instructions to the PDF
c.setFont("Helvetica", 10)
text = c.beginText(40, height - 360 - 20)
text.textLines(instructions)
c.drawText(text)

c.save()

print("User guide has been written to output.pdf")
