
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

app = FastAPI(title="Food Rescue AI Agent")


# Demo NGO and volunteer information
RECIPIENTS = [
    {
        "name": "Hope Community Kitchen",
        "location": "Hyderabad",
        "type": "NGO"
    },
    {
        "name": "Helping Hands Volunteer Group",
        "location": "Secunderabad",
        "type": "Volunteer Group"
    },
    {
        "name": "Green Plate Foundation",
        "location": "Madhapur, Hyderabad",
        "type": "NGO"
    }
]


# Donation details received from the user
class Donation(BaseModel):
    food_type: str
    quantity: int = Field(gt=0)
    prepared_hours_ago: float = Field(ge=0)
    storage: str
    hours_until_expiry: float = Field(ge=0)
    pickup_location: str


def get_response(result):
    messages = result.get("messages", [])

    if messages:
        return messages[-1].content

    return "No response generated."


# Create two AI agents
def build_agents():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing in Render Environment Variables.")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        thinking_level="low",
        timeout=60,
        max_retries=1
    )

    # Agent 1: Food Evaluation
    evaluation_agent = create_agent(
        model=model,
        tools=[],
        system_prompt="""
        You are a Food Safety and Evaluation Agent.

        Evaluate donated food using:
        - Food type
        - Quantity
        - Preparation time
        - Storage method
        - Expiry time

        Explain the urgency of the donation.
        Never guarantee that food is safe based only on the
        information provided. Recommend proper food safety checks.
        """
    )

    # Agent 2: NGO Matching
    matching_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=f"""
        You are an NGO and Volunteer Matching Agent.

        Match food donations with suitable recipients.

        Available demo recipients:
        {RECIPIENTS}

        Consider the pickup location and recipient location.

        Clearly state that these are demo matches,
        not verified NGO contacts.
        Pickup is not confirmed.
        """
    )

    return evaluation_agent, matching_agent


@app.get("/")
def home():
    return {
        "message": "Food Rescue AI Agent is running!",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rescue")
def rescue_food(donation: Donation):

    try:
        evaluation_agent, matching_agent = build_agents()

        donation_details = donation.model_dump()

        # Run Agent 1
        evaluation_result = evaluation_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Evaluate this food donation: {donation_details}"
                }
            ]
        })

        food_evaluation = get_response(evaluation_result)

        # Run Agent 2
        matching_result = matching_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Match this food donation with suitable recipients: "
                        f"{donation_details}. "
                        f"Food evaluation: {food_evaluation}"
                    )
                }
            ]
        })

        ngo_matching = get_response(matching_result)

        return {
            "donation": donation_details,
            "food_evaluation": food_evaluation,
            "ngo_matching": ngo_matching,
            "notice": "Recipient matches are for demonstration only. "
                      "Food safety and pickup must be independently verified."
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

