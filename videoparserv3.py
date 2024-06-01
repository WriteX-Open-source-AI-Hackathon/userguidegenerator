import cv2
import base64
import os
import numpy as np
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO

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

# Print the result in chunks to avoid truncation in the console
chunk_size = 1000
for i in range(0, len(instructions), chunk_size):
    print(instructions[i:i + chunk_size])

# Create a PDF document
pdf_file = "output.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter
c.setFont("Helvetica", 16)

# Add the title to the PDF
c.drawString(40, height - 40, title)

# Add the first frame to the PDF
first_frame = base64Frames[0]
img_data = base64.b64decode(first_frame)
image = ImageReader(BytesIO(img_data))
c.drawImage(image, 40, height - 340, width=512, height=288)

# Add the instructions to the PDF
c.setFont("Helvetica", 12)
text = c.beginText(40, height - 360 - 20)
text.textLines(instructions)
c.drawText(text)

c.save()

print("User guide has been written to output.pdf")
