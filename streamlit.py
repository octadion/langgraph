import streamlit as st
import time
import traceback
from app import super_graph
from langchain_core.messages.human import HumanMessage
# import langchain
# langchain.debug = True
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def main():
    st.title('Chat')

    user_input = st.chat_input("Your message")
    if user_input:
        start = time.time()
        try:
            result = ""
            for s in super_graph.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ],
                },
                {"recursion_limit": 50},
            ):
                if "__end__" not in s:
                    st.write(f"Receiving: {s}")
                    st.write("---")
                    item = next(iter(s.values()))
                    if "messages" in item:
                        result = item["messages"][-1].content
            st.session_state['chat_history'].append({"user": user_input, "assistant": result})
            st.write(time.time() - start)
        except Exception as e:
            st.write(f"Exception: {e}")
            st.write(f"Traceback: {traceback.format_exc()}")

    for chat in st.session_state['chat_history']:
        with st.chat_message(chat["user"]):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["assistant"])

if __name__ == "__main__":
    main()
