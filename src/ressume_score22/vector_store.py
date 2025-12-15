# src/vector_store.py

import os
import pandas as pd
from preprocess import preprocess_text
from embedder import TextEmbedder
import chromadb
from chromadb.config import Settings

class ResumeVectorStore:
    def __init__(self, db_path: str = "./vector_store", collection_name: str = "resume_collection"):
        """
        Initialize Chroma vector store.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.db_path
        ))
        # Create or get collection
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=self.collection_name)
        else:
            self.collection = self.client.create_collection(name=self.collection_name)
        
        self.embedder = TextEmbedder()

    def load_and_preprocess(self, csv_folder: str, text_column: str = "Resume_str") -> pd.DataFrame:
        """
        Load all CSVs in a folder, preprocess the text column, and return dataframe.
        """
        csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
        df_list = []
        for file in csv_files:
            path = os.path.join(csv_folder, file)
            df = pd.read_csv(path)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        df['clean_text'] = df[text_column].astype(str).apply(preprocess_text)
        return df

    def build_vector_store(self, df: pd.DataFrame):
        """
        Generate embeddings and store them in Chroma.
        """
        embeddings = self.embedder.generate_batch_embeddings(df['clean_text'].tolist())
        self.collection.add(
            documents=df['clean_text'].tolist(),
            embeddings=embeddings.tolist(),
            metadatas=df.to_dict(orient="records"),
            ids=[str(i) for i in range(len(df))]
        )
        self.client.persist()
        print(f"Vector store created with {len(df)} documents at {self.db_path}")

    def query(self, query_text: str, top_k: int = 3) -> dict:
        """
        Query the vector store for top_k similar documents.
        """
        query_embedding = self.embedder.generate_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results


# ---------------- Test Run ----------------
if __name__ == "__main__":
    folder_path = r"E:\Ressumme_nlp\archive\Resume"  # your CSV folder
    vector_store = ResumeVectorStore(db_path="./vector_store")

    # Load and preprocess resumes
    df = vector_store.load_and_preprocess(folder_path)

    # Build the vector store
    vector_store.build_vector_store(df)

    # Example query
    query = "Looking for Python Data Science resume"
    results = vector_store.query(query, top_k=3)
    print("\nTop 3 similar resumes:\n", results)
