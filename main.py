import os
import discord
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ——— Load secrets ———
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN       = os.getenv("HF_TOKEN")

# ——— Set up Discord ———
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ——— Set up Hugging Face Inference ———
hf = InferenceClient(token=HF_TOKEN)
MODEL_ID = "tiiuae/falcon-7b-instruct"  # free-to-use instruction-tuned LLM

@client.event
async def on_ready():
    print(f"Siriwolf online as {client.user}")

@client.event
async def on_message(msg):
    if msg.author.bot:
        return
    if msg.content.startswith("!ask"):
        prompt = msg.content[5:].strip()
        # tell them it’s thinking
        await msg.channel.trigger_typing()
        # call the model
        response = hf.text_generation(
            MODEL_ID,
            inputs=prompt,
            parameters={"max_new_tokens":256, "temperature":0.7}
        )
        await msg.channel.send(response.generated_text)

client.run(DISCORD_TOKEN)