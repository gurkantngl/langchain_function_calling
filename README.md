# Langchain Fonksiyon Çağırma ile Müşteri Destek Chatbot'u

Bu proje, Langchain'in fonksiyon çağırma özelliğini kullanarak yaygın müşteri destek taleplerini otomatikleştiren bir chatbot örneğidir. Chatbot, kullanıcıların doğal dildeki isteklerini anlayıp ilgili (mock) API'ları çağırarak yanıt verir.

## Temel Özellikler

*   **Doğal Dil Anlama:** Kullanıcıların sipariş durumu sorgulama, kullanıcı bilgilerini güncelleme, randevu alma ve mağaza bulma gibi taleplerini anlar.
*   **Fonksiyon Çağırma:** Kullanıcının isteğine göre uygun API fonksiyonunu belirleyip çağırır.
*   **API Simülasyonu:** Gerçek API'lar yerine mock fonksiyonlar kullanarak temel işlevselliği gösterir.
*   **Genişletilebilirlik:** Yeni fonksiyonlar ve özellikler eklemek kolaydır.
*   **LLM Entegrasyonu:** Groq API (örneğin, Llama 3) kullanarak dil anlama ve fonksiyon çağırma kararlarını alır.

## Bileşenler

1.  **Mock API Fonksiyonları (`chatbot.py`):**
    *   `get_order_status(order_id: str)`: Sipariş durumunu döndürür.
    *   `update_user_email(user_id: str, new_email: str)`: E-posta adresini günceller.
    *   `schedule_appointment(service_type: str, preferred_date: str, preferred_time: str)`: Randevu oluşturur.
    *   `find_nearest_store(location: str)`: En yakın mağazayı bulur.
2.  **Langchain Araçları (`chatbot.py`):**
    *   Yukarıdaki Python fonksiyonları `@tool` dekoratörü ile Langchain araçlarına dönüştürülmüştür.
3.  **Langchain Agent (`chatbot.py`):**
    *   `create_openai_functions_agent` kullanılarak oluşturulmuş, Groq ile uyumlu bir agent.
    *   Kullanıcı girdisini alır, hangi aracı çağıracağına karar verir, parametreleri çıkarır ve aracı çalıştırır.
4.  **Agent Executor (`chatbot.py`):**
    *   Agent'ın karar alma ve araç çalıştırma döngüsünü yönetir.
5.  **Konfigürasyon (`.env`):**
    *   `GROQ_API_KEY` gibi hassas bilgileri saklar.
6.  **Bağımlılıklar (`requirements.txt`):**
    *   Projenin ihtiyaç duyduğu Python kütüphanelerini listeler.

## Kurulum ve Çalıştırma

1.  **Depoyu Klonlayın (veya dosyaları indirin).**
2.  **Sanal Ortam Oluşturun (Önerilir):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```
3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Anahtarını Ayarlayın:**
    *   `.env` dosyasını oluşturun (veya kopyalayın).
    *   Dosyanın içine kendi Groq API anahtarınızı ekleyin:
        ```
        GROQ_API_KEY='gsk_...'
        ```
5.  **Chatbot'u Çalıştırın:**
    ```bash
    python chatbot.py
    ```
6.  **Etkileşim:** Terminalde chatbot'a sorularınızı veya komutlarınızı yazın. Çıkmak için `quit` yazın.

