
Run env
conda activate ./.conda/simple-rag

docker compose up -d

docker exec -it ollama ollama pull llama3

docker exec -it ollama_rag ollama pull phi3:mini

python rag_pdf.py
