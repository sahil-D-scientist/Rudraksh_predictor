import swisseph as swe
import datetime
from timezonefinder import TimezoneFinder
import pytz
import os

from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
from langchain.chains import create_retrieval_chain
from geopy.geocoders import Nominatim
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_janam_details(year, month, day, hour, minute, second, latitude, longitude):
    """
    Calculate Janam Rashi and Nakshatra using automatic timezone detection.

    Args:
        year (int): Birth year
        month (int): Birth month
        day (int): Birth day
        hour (int): Hour (24-hour format)
        minute (int): Minute
        second (int): Second
        latitude (float): Latitude of birth location
        longitude (float): Longitude of birth location

    Returns:
        dict: Dictionary with Moon Sidereal Longitude, Rashi, Nakshatra, and Pada
    """
    
    # Step 1: Get timezone name using coordinates
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
    if timezone_str is None:
        raise ValueError("Could not determine timezone for given coordinates.")
    
    # Step 2: Localize time and convert to UTC
    local_tz = pytz.timezone(timezone_str)
    local_dt = datetime.datetime(year, month, day, hour, minute, second)
    local_dt = local_tz.localize(local_dt)
    utc_dt = local_dt.astimezone(pytz.utc)

    # Step 3: Calculate Julian Day in UTC
    jd = swe.julday(
        utc_dt.year,
        utc_dt.month,
        utc_dt.day,
        utc_dt.hour + utc_dt.minute / 60 + utc_dt.second / 3600
    )

    # Step 4: Set Lahiri Ayanamsa and compute Moon's position
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    moon_long = swe.calc_ut(jd, swe.MOON)[0][0]
    ayanamsa = swe.get_ayanamsa_ut(jd)
    moon_sidereal = (moon_long - ayanamsa) % 360

    # Step 5: Determine Rashi (Moon Sign)
    rashis = [
        "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)", "Karka (Cancer)",
        "Simha (Leo)", "Kanya (Virgo)", "Tula (Libra)", "Vrischika (Scorpio)",
        "Dhanu (Sagittarius)", "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
    ]
    rashi_index = int(moon_sidereal // 30)

    # Step 6: Determine Nakshatra and Pada
    nakshatras = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu",
        "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta",
        "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha",
        "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
        "Uttara Bhadrapada", "Revati"
    ]
    nakshatra_index = int(moon_sidereal // (360 / 27))
    nakshatra_name = nakshatras[nakshatra_index]

    nakshatra_degree = moon_sidereal % (360 / 27)
    pada = int(nakshatra_degree // (360 / 108)) + 1

    # Return result
    return {
        "rashi": rashis[rashi_index],
        "nakshatra": nakshatra_name,
    }





def get_lat_longi(location_name):
    """
    Get latitude and longitude for a given location name.

    Args:
        location_name (str): Place name (e.g., "Delhi, India")

    Returns:
        tuple: (latitude, longitude)
    """
    geolocator = Nominatim(user_agent="janam-rashi-finder")
    location = geolocator.geocode(location_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        raise ValueError("Location not found")





def create_retrieval_chain_from_faiss(faiss_index):
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("[✅] FAISS retriever created.",retriever)

    prompt_template = ChatPromptTemplate.from_template("""
    You are a Vedic astrology and Rudraksha recommendation expert.

    Your task is to:
    - Analyze user input such as Date of Birth, Place of Birth, Name, Rashi, and Nakshatra.
    - Recommend the most suitable Rudraksha(s).
    - Retrieve the contextual meaning, spiritual benefits, and significance of the recommended Rudraksha(s) using the given context (from a knowledge base / vector DB).

    ---

    User Input:
    {input}

    ---

    Context:
    {context}

    ---

    Provide a detailed response including:
    - Rudraksha type(s) with reasoning
    - Spiritual and astrological relevance
    - How it supports the user's planetary or karmic conditions
    """)

    llm = ChatOpenAI(
    model="gpt-4o",  # Use 'model' instead of 'model_name'
    temperature=0.3
)
    
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
    

def load_faiss(faiss_path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)





def build_user_query(user_info: dict) -> str:
    return (
        f"Suggest a suitable Rudraksha or astrological advice for:\n"
        f"Name: {user_info['name']}\n"
        f"Date of Birth: {user_info['dob']}\n"
        f"Time of Birth: {user_info['tob']}\n"
        f"Birth Place: {user_info['birth_place']}\n"
        f"Rashi: {user_info['rashi']}\n"
        f"Nakshatra: {user_info['nakshatra']}"
    )