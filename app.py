import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
st.set_page_config(
    page_title="Real Estate Listing Description Generator",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

api = st.sidebar.text_input("Enter Your OPENAI API KEY HERE",type="password")


st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Real Estate Listing Description Generator🏙️")
st.sidebar.markdown("### Welcome to the Lyzr Real Estate Listing Description Generator!")
st.markdown(
    "This app uses Lyzr Automata Agent to Generate Real Estate Listing Description based on Project Name,Size,Locality and other details")
if api:
    open_ai_text_completion_model = OpenAIModel(
        api_key=api,
        parameters={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
            "max_tokens": 1500,
        },
    )
else:
    st.sidebar.error("Please Enter Your OPENAI API KEY")

name = st.text_input("Project Name", placeholder="Lodha World View")
size_range = st.slider("Size Range", 100, 10000, (800, 1800))
project_size = st.text_input("Project Size", placeholder="2 Building - 56 Units")
avg_price = st.number_input("Average Price", placeholder="6.81 K/sq.ft")
configuration = st.text_input("Configuration", placeholder="2,3 BHK")
amenties = st.text_input("Amenities", placeholder="Children's Play Meditation Area,Community Hall,Senior Citizen Siteout,Lift(s),Gymnasium")
locality = st.text_input("Locality", placeholder="Bandra West")


def re_description(project_name, size_range, project_size, avg_price, configuration, locality):
    listing_agent = Agent(
        role="Real Estate expert",
        prompt_persona=f"You are an Expert Real Estate Expert and Copywriter too.Your Task Is to write SEO Friendly Real Estate Listing Description based on Input Values."
    )

    prompt = f"""
    Your Task is to generate SEO friendly Real Estate Listing Description using Below Inputs:
    Project Name: {project_name}
    Size Range: {size_range}
    Project Size: {project_size}
    Average Price: {avg_price}
    Configurations: {configuration}
    Amenities: {amenties}
    Locality: {locality}
    
    Step 1: Research About Locality and their nearest features which benefits Property like nearest school,hospital and etc.
    Step 2: Write An SEO friendly Real Estate Listing Description using Above information and Inputs. 
    
    Output:
    Project Description:
    A luxurious residential project with 2bhk homes! Thoughtfully planned to bestow a lavish lifestyle, Kedar Darshan is designed by urban architects and engineers keeping in line with the Vastu principles, striking a perfect balance of space, aesthetics, and amenities. Enjoy an unparalleled experience of urban living with serene surroundings and facilities galore only at Kedar Darshan!
    
    Amenities:
    Modern Living: A suite of world-class amenities catering to every age and interest.
    Children's Play Area: Dedicated space for children to play and enjoy.
    Yoga/Meditation Area: Area for adults to find solace and rejuvenation.
    Grand Entrance Lobby: Exudes grandeur, welcoming residents and guests into a realm of luxury.
    Community Hall: Perfect venue for celebrations and gatherings, fostering a sense of community.
    Senior Citizens' Sit-Out Area: Exclusive area ensuring peace and tranquility for seniors.
    Vastu Compliance: Ensures positive energy flows throughout the project.
    State-of-the-Art Lifts: Equipped for ease of access across all floors.
    
    Features:
    Strategic Location: Proximity to Mumbai's best educational institutions, healthcare facilities, shopping destinations, and entertainment hotspots.
    Educational Institutions: Renowned schools like Podar International School nearby.
    Healthcare Facilities: Hospitals like Global Hospital within reach.
    Connectivity: Well-connected to other parts of Mumbai, offering a blend of luxury and convenience.
    """

    listing_task = Task(
        name="Generate Listing Description",
        model=open_ai_text_completion_model,
        agent=listing_agent,
        instructions=prompt,
    )

    output = LinearSyncPipeline(
        name="Listing Description Pipline",
        completion_message="Product Description Generated!!",
        tasks=[
            listing_task
        ],
    ).run()

    answer = output[0]['task_output']

    return answer


if api and st.button("Generate", type="primary"):
    result = re_description(name,size_range,project_size,avg_price,configuration,locality)
    st.markdown(result)

