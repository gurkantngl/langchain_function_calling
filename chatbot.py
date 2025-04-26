import os
import logging # Ensure logging is imported
from dotenv import load_dotenv
from langchain_groq import ChatGroq # Added import
from langchain.agents import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") # Added
if not groq_api_key:
    print("Uyarı: GROQ_API_KEY ortam değişkeni bulunamadı.")
    # You might want to exit or handle this case differently

# --- Mock API Functions ---

@tool
def get_order_status(order_id: str) -> dict:
    """Verilen sipariş ID'sine göre sipariş durumunu döndürür."""
    try:
        print(f"--- API Çağrısı: get_order_status(order_id={order_id}) ---")
        # Mock data
        statuses = {
            "12345": "Kargoda",
            "67890": "Teslim Edildi",
            "11223": "Hazırlanıyor"
        }
        status = statuses.get(str(order_id), "Sipariş bulunamadı") # Ensure order_id is treated as string key
        return {"order_id": order_id, "status": status, "success": True} # Added success flag
    except Exception as e:
        logging.error(f"get_order_status içinde hata: {e}", exc_info=True) # Log error
        return {"order_id": order_id, "success": False, "error": f"Sipariş durumu alınırken beklenmedik bir hata oluştu: {e}"}


@tool
def update_user_email(user_id: str, new_email: str) -> dict:
    """Belirtilen kullanıcının e-posta adresini günceller ve başarı durumunu döndürür."""
    try:
        print(f"--- API Çağrısı: update_user_email(user_id={user_id}, new_email={new_email}) ---")
        # Mock logic
        if not isinstance(new_email, str) or "@" not in new_email: # Added type check
            return {"user_id": user_id, "success": False, "message": "Geçersiz e-posta adresi formatı."}
        # Assume update is successful
        return {"user_id": user_id, "success": True, "message": f"{user_id} için e-posta {new_email} olarak güncellendi."}
    except Exception as e:
        logging.error(f"update_user_email içinde hata: {e}", exc_info=True)
        return {"user_id": user_id, "success": False, "error": f"E-posta güncellenirken beklenmedik bir hata oluştu: {e}"}


@tool
def schedule_appointment(service_type: str, preferred_date: str, preferred_time: str) -> dict:
    """Belirtilen hizmet için uygunluk kontrolü yapar ve randevu oluşturmaya çalışır, sonucu döndürür."""
    try:
        print(f"--- API Çağrısı: schedule_appointment(service_type={service_type}, preferred_date={preferred_date}, preferred_time={preferred_time}) ---")
        # Mock logic - check basic availability (e.g., not on Sunday)
        if not all(isinstance(arg, str) for arg in [service_type, preferred_date, preferred_time]): # Added type check
             return {"success": False, "message": "Randevu bilgileri metin formatında olmalıdır."}
        if "pazar" in preferred_date.lower():
             return {"success": False, "message": f"{preferred_date} Pazar günü randevu alınamaz."}
        # Assume appointment is scheduled
        appointment_id = f"APT{hash(service_type + preferred_date + preferred_time) % 10000}"
        return {
            "success": True,
            "message": f"{service_type} için {preferred_date} {preferred_time} tarihine randevu oluşturuldu.",
            "appointment_id": appointment_id
        }
    except Exception as e:
        logging.error(f"schedule_appointment içinde hata: {e}", exc_info=True)
        return {"success": False, "error": f"Randevu oluşturulurken beklenmedik bir hata oluştu: {e}"}


@tool
def find_nearest_store(location: str) -> dict:
    """Verilen konuma en yakın mağazanın bilgilerini döndürür."""
    try:
        print(f"--- API Çağrısı: find_nearest_store(location={location}) ---")
        if not isinstance(location, str): # Added type check
            return {"location": location, "success": False, "error": "Konum bilgisi metin formatında olmalıdır."}
        # Mock data based on location
        stores = {
            "ankara": {"name": "Ankara Merkez Mağazası", "address": "Kızılay Cad. No: 1"},
            "istanbul": {"name": "İstanbul Şube", "address": "Taksim Meydanı No: 5"},
            "izmir": {"name": "İzmir Bornova", "address": "Bornova Sok. No: 10"}
        }
        store_info = stores.get(location.lower(), {"name": "Yakın mağaza bulunamadı", "address": ""})
        return {"location": location, "store": store_info, "success": True} # Added success flag
    except Exception as e:
        logging.error(f"find_nearest_store içinde hata: {e}", exc_info=True)
        return {"location": location, "success": False, "error": f"Mağaza bulunurken beklenmedik bir hata oluştu: {e}"}


# --- Langchain Setup ---
tools = [get_order_status, update_user_email, schedule_appointment, find_nearest_store]

# Optional: Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen müşteri hizmetleri taleplerini karşılayan bir asistansın. Uygun aracı çağırarak kullanıcıya yardımcı ol."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Choose the LLM
# Make sure GROQ_API_KEY is set in your environment or .env file
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key) # Added Groq LLM (example model)

# Create the agent
agent = create_openai_functions_agent(llm, tools, prompt) # Re-confirming this works with Groq

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Main Chatbot Logic ---
if __name__ == "__main__":
    print("Chatbot başlatılıyor (Groq ile)...") # Updated message
    print("Çıkmak için 'quit' yazın.")

    chat_history = []

    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'quit':
            break

        if not groq_api_key:
             print("Chatbot: Groq API anahtarı ayarlanmadığı için devam edilemiyor.") # Updated message
             break

        # Invoke the agent
        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            # Print the response
            print("Chatbot:", result['output'])
            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=result['output']))
        except Exception as e:
            print(f"Bir hata oluştu: {e}")
            # Optionally add error handling or logging here

    print("Chatbot kapatıldı.")
