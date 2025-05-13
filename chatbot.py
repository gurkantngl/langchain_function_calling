import os
import logging
import datetime
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool, StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents.format_scratchpad.tools import format_to_tool_messages

# Basit loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock API Fonksiyonları
@tool
def get_order_status(order_id: str) -> Dict[str, Any]:
    """Verilen sipariş numarası için sipariş durumunu sorgular.

    Args:
        order_id: Sorgulanacak sipariş numarası.

    Returns:
        Dict: Sipariş durum bilgilerini içeren sözlük.
    """
    logging.info(f"--- API Çağrısı: get_order_status(order_id={order_id}) ---")
    # Mock yanıt
    order_statuses = {
        "123456": {"status": "Hazırlanıyor", "estimated_delivery": "2023-03-25"},
        "867530": {"status": "Kargoya Verildi", "estimated_delivery": "2023-03-22", 
                 "tracking_number": "TR123456789"},
        # Varsayılan durum
        "default": {"status": "Bulunamadı", "message": "Bu sipariş numarası sistemde bulunamadı."}
    }
    
    result = order_statuses.get(order_id, order_statuses["default"])
    logging.info(f"get_order_status sonucu: {result}")
    return result

@tool
def update_user_email(new_email: str) -> Dict[str, Any]:
    """Kullanıcının email adresini günceller.

    Args:
        new_email: Güncellenecek yeni email adresi.

    Returns:
        Dict: İşlem sonucunu içeren sözlük.
    """
    logging.info(f"--- API Çağrısı: update_user_email(new_email={new_email}) ---")
    
    # Email formatı basit doğrulama
    if "@" not in new_email or "." not in new_email:
        result = {"success": False, "message": "Geçerli bir email adresi girilmelidir."}
    else:
        # Mock başarılı yanıt
        result = {"success": True, "message": f"Email adresiniz {new_email} olarak güncellenmiştir."}
    
    logging.info(f"update_user_email sonucu: {result}")
    return result

@tool
def schedule_appointment(service_type: Optional[str] = None, 
                        preferred_date: Optional[str] = None, 
                        preferred_time: Optional[str] = None) -> Dict[str, Any]:
    """Servis randevusu oluşturur.

    Args:
        service_type: Randevu türü (opsiyonel, belirtilmezse genel servis olarak alınır)
        preferred_date: Tercih edilen tarih (YYYY-MM-DD formatında)
        preferred_time: Tercih edilen saat (HH:MM formatında)

    Returns:
        Dict: Randevu bilgilerini içeren sözlük.
    """
    logging.info(f"--- API Çağrısı: schedule_appointment(service_type={service_type}, preferred_date={preferred_date}, preferred_time={preferred_time}) ---")
    
    # Tarih işleme - "yarın" gibi ifadeleri işle
    try:
        if not preferred_date or preferred_date.lower() == "yarın":
            # Yarın için tarih hesapla
            tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
            preferred_date = tomorrow.strftime("%Y-%m-%d")
            logging.info(f"Tarih 'yarın' olarak belirtilmiş, hesaplanan tarih: {preferred_date}")
        
        # Tarih formatını doğrula ve düzelt
        if preferred_date:
            try:
                parsed_date = datetime.datetime.strptime(preferred_date, "%Y-%m-%d") 
                preferred_date = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                logging.warning(f"Geçersiz tarih formatı: {preferred_date}, varsayılan olarak yarın kullanılıyor")
                tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
                preferred_date = tomorrow.strftime("%Y-%m-%d")
    
        # Servis türü kontrolü
        if not service_type:
            service_type = "Genel Servis"
        
        # Zaman kontrolü
        if not preferred_time:
            preferred_time = "09:00"
            
        # Mock başarılı yanıt - şimdi gerçek tarihi kullanıyor
        result = {
            "success": True, 
            "appointment_id": "APT" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "service_type": service_type,
            "date": preferred_date,
            "time": preferred_time,
            "message": f"{preferred_date} tarihinde saat {preferred_time} için {service_type} randevunuz oluşturulmuştur."
        }
    except Exception as e:
        logging.error(f"Randevu oluşturulurken hata: {str(e)}")
        result = {"success": False, "message": f"Randevu oluşturulamadı: {str(e)}"}
    
    logging.info(f"schedule_appointment sonucu: {result}")
    return result

@tool
def find_nearest_store(location: Optional[str] = None) -> Dict[str, Any]:
    """Verilen konuma en yakın mağazayı bulur.

    Args:
        location: Kullanıcı konumu (şehir adı veya semt)

    Returns:
        Dict: En yakın mağaza bilgisini içeren sözlük.
    """
    logging.info(f"--- API Çağrısı: find_nearest_store(location={location}) ---")
    
    # Konum belirlenmemişse veya geçersizse
    if not location or location.lower() in ["your current location", "my location", "current location"]:
        logging.warning(f"Konum belirtilmemiş veya geçersiz: {location}")
        return {
            "message": "Konum bilgisi alınamadı. Lütfen bulunduğunuz şehir veya semti belirtin.",
            "stores_available": ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]
        }
    
    # Mock mağaza bilgileri
    stores = {
        "istanbul": {
            "name": "İstanbul Merkez Mağaza", 
            "address": "Bağdat Caddesi No:123, Kadıköy",
            "phone": "0212 555 1234",
            "working_hours": "09:00-22:00"
        },
        "ankara": {
            "name": "Ankara Kızılay Mağazası", 
            "address": "Atatürk Bulvarı No:456, Kızılay",
            "phone": "0312 555 5678",
            "working_hours": "10:00-21:00"
        },
        "izmir": {
            "name": "İzmir Karşıyaka Mağazası", 
            "address": "Cemal Gürsel Cad. No:789, Karşıyaka",
            "phone": "0232 555 9012",
            "working_hours": "10:00-22:00"
        },
        "bursa": {
            "name": "Bursa Nilüfer Mağazası", 
            "address": "FSM Bulvarı No:101, Nilüfer",
            "phone": "0224 555 3456",
            "working_hours": "10:00-21:00"
        },
        "antalya": {
            "name": "Antalya Merkez Mağazası", 
            "address": "Konyaaltı Cad. No:202, Merkez",
            "phone": "0242 555 7890",
            "working_hours": "09:00-22:00"
        }
    }
    
    # Konum eşleştirme (basit, türkçe karakter duyarlılığı olmadan)
    normalized_location = location.lower().replace('ı', 'i').replace('ö', 'o').replace('ü', 'u').replace('ş', 's').replace('ğ', 'g').replace('ç', 'c')
    
    for city, store_info in stores.items():
        if city in normalized_location or normalized_location in city:
            result = {
                "success": True,
                "store": store_info,
                "distance": "Yaklaşık 2.5 km",
                "message": f"Size en yakın mağazamız: {store_info['name']}, {store_info['address']}"
            }
            logging.info(f"find_nearest_store sonucu: {result}")
            return result
    
    # Eşleşme bulunamadı
    result = {
        "success": False,
        "message": f"'{location}' konumunda mağaza bulunamadı. Hizmet verdiğimiz şehirler: İstanbul, Ankara, İzmir, Bursa, Antalya",
        "stores_available": ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]
    }
    
    logging.info(f"find_nearest_store sonucu: {result}")
    return result

def main():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Araçların oluşturulması
    tools = [get_order_status, update_user_email, schedule_appointment, find_nearest_store]

    # Prompt şablonu
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen müşteri hizmetleri taleplerini karşılayan bir asistansın. Uygun aracı çağırarak kullanıcıya yardımcı ol."),
        MessagesPlaceholder(variable_name="chat_history", optional=True), 
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatGroq(
        model="llama3-8b-8192", 
        groq_api_key=groq_api_key
    )

    # Agent oluşturulması
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )

    # Terminal tabanlı konuşma döngüsü
    chat_history = []
    print("🤖 Müşteri Destek Botuna Hoş Geldiniz! (Çıkmak için 'exit' yazın)")

    while True:
        user_input = input("\n🧑‍💻 Siz: ")
        if user_input.lower() in ["exit", "quit", "çıkış"]:
            print("👋 Görüşmek üzere!")
            break

        # Agent'a istek gönder
        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            response = result["output"]
            print(f"\n🤖 Bot: {response}")

            # Konuşma geçmişini güncelle
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            
            # Geçmiş çok uzarsa en eskilerini çıkar
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
                
        except Exception as e:
            logging.exception("Agent çalıştırılırken bir hata oluştu")
            print(f"\n🤖 Bot: Üzgünüm, bir sorun oluştu: {str(e)}")

if __name__ == "__main__":
    main()
