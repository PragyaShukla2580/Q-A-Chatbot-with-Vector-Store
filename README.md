# Q-A-Chatbot-with-Vector-Store
Two apis, one to store the chunks of all the documents in the folder and initiate chat history in Faiss index and another to query (RAG) over the stored chunks.

# Demo images

**1. Payload for storing chunks and initializing chat history in Faiss index.**

![image](https://github.com/user-attachments/assets/d9fa4ac7-2706-40bf-94a1-25cbd7364b5c)


**2. Output for start faiss.**

![image](https://github.com/user-attachments/assets/cba4b55e-cb32-4262-a2f8-f98ad8c7033d)


**3. Query: How can I build an ubuntu Image? -- start**

![image](https://github.com/user-attachments/assets/53a465b6-9d62-4521-bc0d-3a0d84c01b54)


**4. Response:**

![image](https://github.com/user-attachments/assets/02e94d54-856e-44f5-afba-2b088cd541c2)


**5. Query: "How can I do it for a specific version?" -- continue**

![image](https://github.com/user-attachments/assets/2ed08e0f-e36a-4559-943f-2d0a42ecb9a3)


**6. Response:**

![image](https://github.com/user-attachments/assets/1842c982-b7e9-4992-96e0-37b9c5a1dd12)

![image](https://github.com/user-attachments/assets/203ecbdd-a899-47e3-bc2b-7c56ec11a0ef)


**7. Query: "Give me the format for account-key assertion." -- start**

![image](https://github.com/user-attachments/assets/482cf4e2-2555-423e-a0bf-b64f2f22dc83)


**8. Response:**

![image](https://github.com/user-attachments/assets/3fa975e7-4d98-48d1-8b01-ba55a7195d56)


**9. stop**

![image](https://github.com/user-attachments/assets/7171495f-d471-4b92-b20a-4049ccd7e80d)


**10. Response:**

![image](https://github.com/user-attachments/assets/982e7a4b-1345-4099-a9d7-8d2cb2054a76)


# Docker Container commands


**1. Building a Docker image for the RAG FastAPI application -----> docker build -t fastapi-rag .**


![image](https://github.com/user-attachments/assets/0844e34a-17c1-4554-aa36-8905215d50c6)


**2. Running a Docker container from the RAG FastAPI image with a mounted volume. -----> docker run -p 8080:8080 -v "./demo_bot_data:/app/data" fastapi-rag**


![image](https://github.com/user-attachments/assets/90df38b1-c880-4ae1-a3d8-173b7a33fea0)


![image](https://github.com/user-attachments/assets/8bfc7046-0a9d-4616-83fc-7b070cdd99ac)


**3. Payload in Docker (http://localhost:8080/docs#/default/query_rag_query_post) ----> Query: How can I build an ubuntu Image? -- start**


![image](https://github.com/user-attachments/assets/0216fb9f-0bce-4a49-8a21-d0bc645b0b9b)


**4. Response:**


![image](https://github.com/user-attachments/assets/182d5121-62e2-44c4-9328-112b4c69b49d)


![image](https://github.com/user-attachments/assets/7eb726aa-6cfb-4331-9809-b34db47add53)


