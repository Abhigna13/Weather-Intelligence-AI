from api.agent.tool_calling_agent import ask_weather_tool_agent


THREAD_ID = "real-memory-test-1"

print("=" * 70)
print("WEATHER AI REAL MULTI-TURN MEMORY TEST")
print("=" * 70)


# -------------------------------
# TURN 1
# -------------------------------

question1 = (
    "Which machine learning model is used for temperature "
    "prediction in this project?"
)

result1 = ask_weather_tool_agent(
    question1,
    thread_id=THREAD_ID
)

print("\nQuestion 1:")
print(question1)

print("\nAnswer 1:")
print(result1["answer"])


# -------------------------------
# TURN 2
# -------------------------------

question2 = "What is the model name you mentioned?"

result2 = ask_weather_tool_agent(
    question2,
    thread_id=THREAD_ID
)

print("\nQuestion 2:")
print(question2)

print("\nAnswer 2:")
print(result2["answer"])


# -------------------------------
# MEMORY CHECK
# -------------------------------

print("\nConversation History After Two Turns:")

for item in result2["conversation_history"]:
    print("\n" + item)


print("\n" + "=" * 70)
print("REAL MULTI-TURN MEMORY TEST COMPLETED")
print("=" * 70)