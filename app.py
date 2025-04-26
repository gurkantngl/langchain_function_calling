import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish # Added imports for typing intermediate steps
import logging

# Import tools from chatbot.py
from chatbot import get_order_status, update_user_email, schedule_appointment, find_nearest_store

# --- Configuration and Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Müşteri Destek Chatbot", page_icon="🤖")
st.title("🤖 Müşteri Destek Chatbot")
st.caption("Langchain & Groq ile Güçlendirilmiştir")

# --- Langchain Agent Setup ---
@st.cache_resource
def load_agent_executor():
    tools = [get_order_status, update_user_email, schedule_appointment, find_nearest_store]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen müşteri hizmetleri taleplerini karşılayan bir asistansın. Uygun aracı çağırarak kullanıcıya yardımcı ol."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
    agent = create_tool_calling_agent(llm, tools, prompt)
    # Set verbose=False, return_intermediate_steps=True
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False, # Set back to False for cleaner terminal
        handle_parsing_errors=True,
        return_intermediate_steps=True # Added to get steps
    )
    return agent_executor

if not groq_api_key:
    logging.error("GROQ_API_KEY ortam değişkeni bulunamadı.")
    st.error("GROQ API Anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin ve uygulamayı yeniden başlatın.")
    st.stop()

try:
    agent_executor = load_agent_executor()
    logging.info("Agent Executor başarıyla yüklendi.")
except Exception as e:
    logging.exception("Agent Executor yüklenirken kritik bir hata oluştu.")
    st.error(f"Agent başlatılırken bir hata oluştu: {e}. Lütfen API anahtarınızı ve yapılandırmayı kontrol edin.")
    st.stop()


# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Interaction ---
if prompt := st.chat_input("Sorunuzu veya talebinizi yazın..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare chat history for the agent
    chat_history_for_agent = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history_for_agent.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history_for_agent.append(AIMessage(content=msg["content"]))

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for final response

        try:
            logging.info(f"Kullanıcı girdisi: {prompt}")
            # Invoke the agent - use the last user message and the history
            # We need to pop the last user message from history before passing
            if chat_history_for_agent and isinstance(chat_history_for_agent[-1], HumanMessage):
                 input_for_agent = chat_history_for_agent.pop().content
            else:
                 input_for_agent = prompt

            # Invoke the agent and get intermediate steps
            result = agent_executor.invoke({
                "input": input_for_agent,
                "chat_history": chat_history_for_agent
            })

            # --- Display Intermediate Steps --- (Added/Modified section)
            if "intermediate_steps" in result and result["intermediate_steps"]:
                # Use st.expander for a collapsible view of steps
                with st.expander("⚙️ Agent Çalışma Adımları", expanded=True):
                    for step in result["intermediate_steps"]:
                        # Ensure step is a tuple (AgentAction, observation)
                        if isinstance(step, tuple) and len(step) == 2:
                            action, observation = step
                            if isinstance(action, AgentAction):
                                tool = action.tool
                                tool_input = action.tool_input
                                st.markdown(f"**🛠️ Araç Çağrıldı:** `{tool}`")
                                st.markdown(f"**📥 Parametreler:**")
                                st.json(tool_input, expanded=False) # Keep parameters collapsed initially
                                st.markdown(f"**🔍 Araç Sonucu:**")
                                st.json(observation, expanded=False) # Keep results collapsed initially
                                st.divider() # Add a separator between steps
                            else:
                                st.write("Beklenmeyen adım formatı (action):", action)
                        else:
                             st.write("Beklenmeyen adım formatı (step):", step)


            # --- Final Response ---
            full_response = result.get('output', "Üzgünüm, bir yanıt oluşturamadım.")
            logging.info(f"Agent yanıtı: {full_response}")

        except Exception as e:
            logging.exception(f"Agent çağrılırken hata oluştu. Kullanıcı girdisi: {prompt}")
            full_response = "Üzgünüm, isteğinizi işlerken bir sorun oluştu. Lütfen tekrar deneyin veya farklı bir şekilde sorun."
            st.error(full_response + f" (Detay: {e})")
            # Ensure error message is displayed even if steps fail
            message_placeholder.markdown(full_response) # Display error in the main placeholder


        # Display the final response *after* the intermediate steps expander
        message_placeholder.markdown(full_response)


    # Add assistant response to chat history (even if it's an error message)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

