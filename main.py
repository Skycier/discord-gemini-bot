import os
import logging
import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from transformers import pipeline

# Attempt to import discord with fallback instructions
try:
    import discord
    from discord.ext import commands
except ImportError:
    print("Error: discord.py is not installed. Please install it with:")
    print("pip install discord.py")
    exit(1)

# === Configuration ===
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AstroVisionBot")

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Load CLIP model
logger.info("Loading CLIP model...")
vision_classifier = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-large-patch14"
)
logger.info("Vision model loaded successfully")

# === Astronomy Labels ===
# All 88 modern constellations according to IAU
CONSTELLATIONS_88 = [
    "Andromeda", "Antlia", "Apus", "Aquarius", "Aquila", "Ara", "Aries", 
    "Auriga", "Boötes", "Caelum", "Camelopardalis", "Cancer", "Canes Venatici",
    "Canis Major", "Canis Minor", "Capricornus", "Carina", "Cassiopeia", "Centaurus",
    "Cepheus", "Cetus", "Chamaeleon", "Circinus", "Columba", "Coma Berenices",
    "Corona Australis", "Corona Borealis", "Corvus", "Crater", "Crux", "Cygnus",
    "Delphinus", "Dorado", "Draco", "Equuleus", "Eridanus", "Fornax", "Gemini",
    "Grus", "Hercules", "Horologium", "Hydra", "Hydrus", "Indus", "Lacerta",
    "Leo", "Leo Minor", "Lepus", "Libra", "Lupus", "Lynx", "Lyra", "Mensa",
    "Microscopium", "Monoceros", "Musca", "Norma", "Octans", "Ophiuchus",
    "Orion", "Pavo", "Pegasus", "Perseus", "Phoenix", "Pictor", "Pisces",
    "Piscis Austrinus", "Puppis", "Pyxis", "Reticulum", "Sagitta", "Sagittarius",
    "Scorpius", "Sculptor", "Scutum", "Serpens", "Sextans", "Taurus", "Telescopium",
    "Triangulum", "Triangulum Australe", "Tucana", "Ursa Major", "Ursa Minor",
    "Vela", "Virgo", "Volans", "Vulpecula"
]

ASTRONOMY_LABELS = {
    "constellations": CONSTELLATIONS_88,
    "stars": [
        "Sirius", "Canopus", "Rigil Kentaurus", "Arcturus", "Vega", "Capella",
        "Rigel", "Procyon", "Achernar", "Betelgeuse", "Hadar", "Altair",
        "Acrux", "Aldebaran", "Antares", "Spica", "Pollux", "Fomalhaut",
        "Deneb", "Mimosa", "Regulus", "Adhara", "Castor", "Gacrux",
        "Shaula", "Bellatrix", "Elnath", "Miaplacidus", "Alnilam", "Alnitak"
    ],
    "deep_sky": [
        "spiral galaxy", "elliptical galaxy", "irregular galaxy", 
        "globular cluster", "open cluster", "planetary nebula",
        "emission nebula", "reflection nebula", "dark nebula",
        "supernova remnant", "galaxy cluster", "quasar"
    ],
    "planets": [
        "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", 
        "Uranus", "Neptune", "Pluto"
    ],
    "moon_phases": [
        "new moon", "waxing crescent", "first quarter", "waxing gibbous",
        "full moon", "waning gibbous", "last quarter", "waning crescent"
    ],
    "solar_system": [
        "Sun", "Moon", "asteroid", "comet", "meteor", "aurora", "zodiacal light"
    ],
    "space_objects": [
        "black hole", "neutron star", "pulsar", "binary star", "exoplanet",
        "brown dwarf", "white dwarf", "red giant", "supergiant"
    ]
}

# Format constellation labels
ASTRONOMY_LABELS["constellations"] = [f"{c} constellation" for c in ASTRONOMY_LABELS["constellations"]]
ASTRONOMY_LABELS["stars"] = [f"{s} star" for s in ASTRONOMY_LABELS["stars"]]

# === Vision Processing ===
async def download_image(url: str) -> Image.Image:
    """Download image asynchronously"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Image download failed (HTTP {response.status})")
                image_data = await response.read()
                return Image.open(BytesIO(image_data)).convert("RGB")
        except asyncio.TimeoutError:
            raise ValueError("Image download timed out")

async def classify_astronomy_image(image_url: str) -> dict:
    """
    Classify astronomical features in an image
    Returns dictionary with classification results
    """
    try:
        image = await download_image(image_url)
        results = {}
        
        # Classify all categories
        for category, labels in ASTRONOMY_LABELS.items():
            # Skip categories with too many labels for performance
            if len(labels) > 100:  # Constellations have 88
                # Process in chunks
                chunk_size = 30
                category_results = []
                for i in range(0, len(labels), chunk_size):
                    chunk = labels[i:i+chunk_size]
                    chunk_results = await asyncio.get_event_loop().run_in_executor(
                        None, vision_classifier, image, chunk
                    )
                    if chunk_results:
                        category_results.extend(chunk_results)
                
                # Find best result from chunks
                if category_results:
                    best_result = max(category_results, key=lambda x: x['score'])
                    results[category] = {
                        "label": best_result['label'],
                        "score": best_result['score']
                    }
            else:
                # Run classification for smaller categories
                category_results = await asyncio.get_event_loop().run_in_executor(
                    None, vision_classifier, image, labels
                )
                if category_results:
                    results[category] = {
                        "label": category_results[0]['label'],
                        "score": category_results[0]['score']
                    }
        
        return results
    
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise

# === Bot Commands ===
@bot.command(name="astro", help="Analyze astronomical features in an image")
async def astro_vision(ctx, image_url: str = None):
    """All-in-one astronomy image analysis"""
    # Get image from attachment or URL
    if not image_url and ctx.message.attachments:
        image_url = ctx.message.attachments[0].url
    
    if not image_url:
        return await ctx.reply("🔭 Please attach an image or provide an image URL!")
    
    await ctx.trigger_typing()
    
    try:
        # Classify image
        results = await classify_astronomy_image(image_url)
        
        # Build response
        response = ["🔭 **Astronomy Vision Analysis**"]
        for category, data in results.items():
            label = data['label']
            score = data['score']
            
            # Format category names
            pretty_category = category.replace('_', ' ').title()
            
            # Add emoji based on category
            emoji = {
                "constellations": "🌌",
                "stars": "⭐",
                "deep_sky": "🌀",
                "planets": "🪐",
                "moon_phases": "🌙",
                "solar_system": "🌞",
                "space_objects": "💫"
            }.get(category, "✨")
            
            # Format confidence as stars
            confidence_stars = "★" * min(5, int(score * 5) + 1)
            
            response.append(
                f"\n{emoji} **{pretty_category}**: {label} "
                f"\n   Confidence: {confidence_stars} {score:.1%}"
            )
        
        # Add footer
        response.append("\n\n_Contains all 88 modern constellations_")
        
        # Send response
        await ctx.reply('\n'.join(response))
        
    except Exception as e:
        logger.error(f"Vision analysis error: {str(e)}")
        await ctx.reply(f"🔴 Vision analysis failed: {str(e)}")

# === Bot Events ===
@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name=f"{len(CONSTELLATIONS_88)} constellations"
        )
    )
    logger.info(f"Vision bot ready with {len(CONSTELLATIONS_88)} constellations")

# === Error Handling ===
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.reply("❌ Unknown command. Use `!astro` with an attached image!")
    else:
        logger.error(f"Command error: {str(error)}")
        await ctx.reply(f"⚠️ Vision processing error: {str(error)}")

# === Main Execution ===
if __name__ == "__main__":
    logger.info("Starting Astronomy Vision Bot...")
    logger.info(f"Loaded {len(CONSTELLATIONS_88)} constellations")
    try:
        bot.run(DISCORD_TOKEN)
    except discord.LoginError:
        logger.critical("Invalid Discord token. Check your .env file!")
    except Exception as e:
        logger.critical(f"Fatal startup error: {str(e)}")