import streamlit as st
import datetime
from utils import (
    get_janam_details,
    get_lat_longi,
    load_faiss,
    create_retrieval_chain_from_faiss,
    build_user_query
)

st.title("🔮 Rudraksha Recommendation Based on Janam Kundli")

# Initialize session state
for key, default in {
    'form_submitted': False,
    'details_confirmed': False,
    'rashi': None,
    'nakshatra': None,
    'details': None,
    'name': '',
    'dob': datetime.date(2000, 1, 1),
    'tob': '04:00',
    'birth_place': '',
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Step 1: Input Form ---
with st.form("birth_form", clear_on_submit=False):
    st.subheader("📝 Enter Your Birth Details")
    st.session_state.name = st.text_input("Your Name", value=st.session_state.name)
    st.session_state.dob = st.date_input("Date of Birth", value=st.session_state.dob,
                                          min_value=datetime.date(1900, 1, 1),
                                          max_value=datetime.date.today())
    st.session_state.tob = st.text_input("Time of Birth (e.g. 04:00)", value=st.session_state.tob)
    st.session_state.birth_place = st.text_input("Place of Birth", value=st.session_state.birth_place)
    submitted = st.form_submit_button("Get Rudraksha Suggestion")

    if submitted:
        try:
            tob_parsed = datetime.datetime.strptime(st.session_state.tob.strip(), "%H:%M").time()
            lat, longi = get_lat_longi(st.session_state.birth_place)
            details = get_janam_details(
                year=st.session_state.dob.year,
                month=st.session_state.dob.month,
                day=st.session_state.dob.day,
                hour=tob_parsed.hour,
                minute=tob_parsed.minute,
                second=0,
                latitude=lat,
                longitude=longi
            )
            st.session_state.rashi = details['rashi']
            st.session_state.nakshatra = details['nakshatra']
            st.session_state.details = details
            st.session_state.form_submitted = True
        except ValueError:
            st.error("⚠️ Invalid time format. Use HH:MM.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# --- Step 2: Confirm Details (Only shown if form is submitted) ---
if st.session_state.form_submitted and not st.session_state.details_confirmed:
    st.divider()
    st.subheader("🪐 Predicted Janam Details")
    st.write(f"**Rashi (Moon Sign):** {st.session_state.rashi}")
    st.write(f"**Nakshatra:** {st.session_state.nakshatra}")

    edit = st.checkbox("✏️ Edit Rashi & Nakshatra manually")

    rashis = [
        "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)", "Karka (Cancer)",
        "Simha (Leo)", "Kanya (Virgo)", "Tula (Libra)", "Vrischika (Scorpio)",
        "Dhanu (Sagittarius)", "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
    ]
    nakshatras = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu",
        "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta",
        "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha",
        "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
        "Uttara Bhadrapada", "Revati"
    ]

    rashi_selected = st.selectbox("Select Rashi", rashis, index=rashis.index(st.session_state.rashi)) if edit else st.session_state.rashi
    nakshatra_selected = st.selectbox("Select Nakshatra", nakshatras, index=nakshatras.index(st.session_state.nakshatra)) if edit else st.session_state.nakshatra

    if st.button("✅ Confirm and Get Suggestion"):
        st.session_state.rashi = rashi_selected
        st.session_state.nakshatra = nakshatra_selected
        st.session_state.details_confirmed = True

# --- Step 3: Show Suggestion ---
if st.session_state.details_confirmed:
    st.divider()
    st.subheader("📋 Final Janam Details Used")
    st.write(f"**Name:** {st.session_state.name}")
    st.write(f"**Date of Birth:** {st.session_state.dob.strftime('%Y-%m-%d')}")
    st.write(f"**Time of Birth:** {st.session_state.tob}")
    st.write(f"**Place of Birth:** {st.session_state.birth_place}")
    st.write(f"**Rashi:** {st.session_state.rashi}")
    st.write(f"**Nakshatra:** {st.session_state.nakshatra}")

    with st.spinner("Fetching your Rudraksha suggestion..."):
        faiss_index = load_faiss("faiss_index_rudra_updated")
        retrieval_chain = create_retrieval_chain_from_faiss(faiss_index)
        result = retrieval_chain.invoke({
            "input": build_user_query({
                "name": st.session_state.name,
                "dob": st.session_state.dob.strftime("%Y-%m-%d"),
                "tob": st.session_state.tob,
                "birth_place": st.session_state.birth_place,
                "rashi": st.session_state.rashi,
                "nakshatra": st.session_state.nakshatra
            })
        })

        st.subheader("📿 Rudraksha / Astrology Suggestion")
        st.markdown(result["answer"])

    if st.button("🔄 Start Over"):
        for key in ['form_submitted', 'details_confirmed', 'rashi', 'nakshatra', 'details']:
            st.session_state[key] = False
