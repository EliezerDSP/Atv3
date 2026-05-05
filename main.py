import ollama

def generate_embeddings():
    # 1. Configurações iniciais
    model_name = "nomic-embed-text"
    text_input = "O aprendizado de máquina está transformando a tecnologia."

    print(f"--- Conectando ao Ollama e processando o texto: '{text_input}' ---\n")

    try:
        # 2. Chamada à API local do Ollama para gerar embeddings
        response = ollama.embeddings(model=model_name, prompt=text_input)
        
        # 3. Extração do vetor
        embedding_vector = response['embedding']
        
        # 4. Exibição dos resultados
        print(f"Sucesso! Vetor gerado.")
        print(f"Tamanho do vetor (dimensões): {len(embedding_vector)}")
        print(f"Primeiros 5 valores do vetor: {embedding_vector[:5]}...")
        
    except Exception as e:
        print(f"Erro ao conectar com o Ollama: {e}")

if __name__ == "__main__":
    generate_embeddings()
