import time
import traceback
from app import super_graph
from langchain_core.messages.human import HumanMessage
# import langchain
# langchain.debug = True
start = time.time()
try:
    result = ""
    for s in super_graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Apakah ada anggota yang berusia di atas 50 tahun?"
                )
            ],
        },
        {"recursion_limit": 50},
    ):
        if "__end__" not in s:
            print(f"Receiving: {s}")
            print("---")
            item = next(iter(s.values()))
            if "messages" in item:
                result = item["messages"][-1].content
    print("The final result: " + result)
    print(time.time() - start)
except Exception as e:
    print(f"Exception: {e}")
    print(f"Traceback: {traceback.format_exc()}")