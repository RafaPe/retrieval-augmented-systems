from simplerag.rag import run_simple_rag


if __name__ == "__main__":
    run_simple_rag(
        gen_model="llama3",
        judge_model="phi3:mini",
        k=3
    )
