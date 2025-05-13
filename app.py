import streamlit as st
import os
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish

# Import tools from chatbot.py
from chatbot import get_order_status, update_user_email, schedule_appointment, find_nearest_store

def main():
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
        
        # Bugün, yarın ve hafta günleri için tarih hesapla
        today = datetime.now()
        
        # Bugün ve yarın için tarih formatı
        today_str = today.strftime("%Y-%m-%d")
        tomorrow = today + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        
        # Hafta günleri için tarih hesaplama
        current_weekday = today.weekday()  # 0: Pazartesi, 1: Salı, ..., 6: Pazar
        
        weekdays = {
            "pazartesi": (0 - current_weekday) % 7 or 7,
            "salı": (1 - current_weekday) % 7 or 7,
            "çarşamba": (2 - current_weekday) % 7 or 7,
            "perşembe": (3 - current_weekday) % 7 or 7,
            "cuma": (4 - current_weekday) % 7 or 7,
            "cumartesi": (5 - current_weekday) % 7 or 7,
            "pazar": (6 - current_weekday) % 7 or 7
        }
        
        # Tarihleri hesapla
        weekday_dates = {}
        for day_name, days_ahead in weekdays.items():
            date = today + timedelta(days=days_ahead)
            weekday_dates[day_name] = date.strftime("%Y-%m-%d")
        
        # Sistem talimatlarını güncelleyerek daha net yönlendirmeler ekliyorum
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen müşteri hizmetleri taleplerini karşılayan bir asistansın. Uygun aracı çağırarak kullanıcıya yardımcı ol.

Fonksiyon çağırma kuralları:
- Sipariş durumu sorularında get_order_status kullan
- E-posta güncelleme için update_user_email kullan
- Randevu oluşturma için schedule_appointment kullan.
- En yakın mağaza sorguları için find_nearest_store kullan ve lokasyon olarak şehir adı belirt
- Kullanıcının lokasyonu yoksa veya "Your Current Location" çağrısı yapıyorsan, bunun yerine lokasyon için kullanıcıdan bilgi iste

Tarih bilgisi:
- Bugünün tarihi: {today_date}
- Yarının tarihi: {tomorrow_date}
- Önümüzdeki/bu/gelecek hafta için tarihler:
  * Pazartesi: {monday_date}
  * Salı: {tuesday_date}
  * Çarşamba: {wednesday_date}
  * Perşembe: {thursday_date}
  * Cuma: {friday_date}
  * Cumartesi: {saturday_date}
  * Pazar: {sunday_date}
- Tüm tarihler YYYY-MM-DD formatında kullanılmalıdır
- "Yarın" ifadesi kullanıldığında {tomorrow_date} tarihini kullan
- "Bugün" ifadesi kullanıldığında {today_date} tarihini kullan
- "Bu pazartesi", "önümüzdeki pazartesi" gibi ifadeler için yukarıdaki tarihleri kullan
- "Önümüzdeki hafta" ifadesinde bir sonraki pazartesi ({monday_date}) olarak kabul et
- Başka tarihleri hesaplarken de YYYY-MM-DD formatını koruyarak hesapla

Geçersiz bir yanıt verme ve her zaman Türkçe konuş.""".format(
                today_date=today_str,
                tomorrow_date=tomorrow_str,
                monday_date=weekday_dates["pazartesi"],
                tuesday_date=weekday_dates["salı"],
                wednesday_date=weekday_dates["çarşamba"],
                thursday_date=weekday_dates["perşembe"],
                friday_date=weekday_dates["cuma"],
                saturday_date=weekday_dates["cumartesi"],
                sunday_date=weekday_dates["pazar"]
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Groq modelini yapılandır ve hata durumuna karşı koruma ekle
        try:
            llm = ChatGroq(
                model="llama3-8b-8192", 
                groq_api_key=groq_api_key,
                temperature=0.1,  # Daha kararlı yanıtlar için düşük sıcaklık
                max_tokens=4000,   # Yeterli token sayısı
                request_timeout=30, # Zaman aşımı süresi (saniye)
            )
        except Exception as e:
            logging.error(f"LLM yapılandırılırken hata: {e}")
            st.error(f"Model yüklenirken hata oluştu: {e}")
            st.stop()

        # Parsing hatalarına karşı koruma ile agent oluştur
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools, 
            verbose=False,  # Üretim ortamında verbose kapatılabilir
            handle_parsing_errors=True,
            max_iterations=3,  # Sonsuz döngü riskini azaltmak için
            return_intermediate_steps=True
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
        
    # Hafıza kısmında herhangi bir mesaj yoksa giriş yazısını göster
    if len(st.session_state.messages) == 0:
        st.markdown("""
        ### 👋 Hoş Geldiniz!
        
        Bu demo, LangChain kullanarak geliştirilmiş bir müşteri destek asistanıdır. Asistan aşağıdaki işlemleri yapabilir:
        
        - **Sipariş durumu sorgulama**: "123456 sipariş numaralı siparişimin durumu nedir?"
        - **E-posta güncelleme**: "E-posta adresimi example@example.com olarak güncellemek istiyorum"
        - **Randevu oluşturma**: "Yarın saat 14:00'te bir servis randevusu almak istiyorum"
        - **En yakın mağazayı bulma**: "İstanbul'da en yakın mağazanız nerede?"
        
        ### Örnek Sorular:
        - "Merhaba, 867530 numaralı siparişimin durumu nedir?"
        - "E-posta adresimi yeni_mail@gmail.com olarak değiştirmek istiyorum."
        - "Yarın öğleden sonra bir servis randevusu ayarlamak istiyorum."
        - "Ankara'da en yakın mağazanız nerede?"
        
        **Bir soru sormak için aşağıdaki metin kutusuna yazın.**
        """)

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
            loading_text = "Yanıt hazırlanıyor..."
            message_placeholder.markdown(loading_text)

            try:
                logging.info(f"Kullanıcı girdisi: {prompt}")
                
                # Yarın için tarihi hesapla (randevu işlemleri için)
                tomorrow = datetime.now() + timedelta(days=1)
                tomorrow_str = tomorrow.strftime("%Y-%m-%d")
                
                # Kullanıcının girdisini olduğu gibi kullan, çünkü sistem promptunda zaten tarih bilgisi var
                enhanced_prompt = prompt
                
                # Invoke the agent and get intermediate steps
                if chat_history_for_agent and isinstance(chat_history_for_agent[-1], HumanMessage):
                    input_for_agent = chat_history_for_agent.pop().content
                else:
                    input_for_agent = enhanced_prompt
                
                # Agent'i çağır ve maksimum 2 deneme yap
                retry_count = 0
                max_retries = 2
                last_error = None
                
                while retry_count <= max_retries:
                    try:
                        # Sadeleştirilmiş agent çağrısı
                        result = agent_executor.invoke({
                            "input": input_for_agent,
                            "chat_history": chat_history_for_agent
                        })
                        
                        # Başarılı çağrıyla döngüyü kır
                        break
                    except Exception as e:
                        last_error = str(e)
                        logging.warning(f"Agent çağrısı hatası (Deneme {retry_count+1}/{max_retries+1}): {e}")
                        retry_count += 1
                        
                        # Son denemedeysek ve hala hata varsa, hatayı fırlatmaya devam et
                        if retry_count > max_retries:
                            raise e
                
                # --- Display Intermediate Steps ---
                if "intermediate_steps" in result and result["intermediate_steps"]:
                    with st.expander("⚙️ Agent Çalışma Adımları", expanded=False):
                        for step in result["intermediate_steps"]:
                            # Ensure step is a tuple (AgentAction, observation)
                            if isinstance(step, tuple) and len(step) == 2:
                                action, observation = step
                                if isinstance(action, AgentAction):
                                    tool = action.tool
                                    tool_input = action.tool_input
                                    st.markdown(f"**🛠️ Araç Çağrıldı:** `{tool}`")
                                    
                                    try:
                                        if isinstance(tool_input, str):
                                            tool_input_dict = json.loads(tool_input)
                                        else:
                                            tool_input_dict = tool_input
                                        st.json(tool_input_dict, expanded=False)
                                    except:
                                        st.code(f"{tool_input}")
                                        
                                    st.markdown(f"**🔍 Araç Sonucu:**")
                                    try:
                                        if isinstance(observation, (dict, list)):
                                            st.json(observation, expanded=False)
                                        else:
                                            st.code(f"{observation}")
                                    except:
                                        st.write(f"{observation}")
                                    st.divider()
                                else:
                                    st.write("Beklenmeyen adım formatı (action):", action)
                            else:
                                st.write("Beklenmeyen adım formatı (step):", step)

                # --- Final Response ---
                full_response = result.get('output', "Üzgünüm, bir yanıt oluşturamadım.")
                logging.info(f"Agent yanıtı: {full_response}")
                message_placeholder.markdown(full_response)

            except Exception as e:
                logging.exception(f"Agent çağrılırken hata oluştu. Kullanıcı girdisi: {prompt}")
                
                # Kullanıcı dostu hata mesajı
                if "Failed to call a function" in str(e):
                    error_message = """Özür dilerim, randevunuzu oluştururken bir sorun yaşadım. 
                    
Lütfen talebinizi şu şekilde belirtin:
- Hangi hizmet için randevu istediğinizi belirtin (örn: "servis randevusu")
- Tarihi belirtin (örn: "yarın", "bugün", "önümüzdeki salı" gibi)
- Saati belirtin (örn: "14:00")

Örnek: "Önümüzdeki salı saat 14:00'te telefon tamiri için randevu almak istiyorum."
                    """
                elif "Your Current Location" in str(e):
                    error_message = "Özür dilerim, konumunuzu belirtmediğiniz için size en yakın mağazayı bulamadım. Lütfen bulunduğunuz şehir veya semti belirtin. Örnek: 'Ankara'da en yakın mağazanız nerede?'"
                else:
                    error_message = f"Üzgünüm, isteğinizi işlerken bir sorun oluştu. Lütfen farklı bir şekilde sorunuzu ifade eder misiniz?"
                
                message_placeholder.markdown(error_message)
                full_response = error_message

        # Add assistant response to chat history (even if it's an error message)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

