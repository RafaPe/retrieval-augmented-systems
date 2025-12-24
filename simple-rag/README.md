
Run env
conda activate ./.conda/simple-rag

docker compose up -d

docker exec -it ollama ollama pull llama3

python rag_pdf.py
