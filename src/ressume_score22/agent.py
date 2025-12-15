# src/agent.py

from vector_store import ResumeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

class ResumeAgent:
    def __init__(self, vector_store_path: str = "./vector_store", collection_name: str = "resume_collection", llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize Resume Agent
        """
        self.vstore = ResumeVectorStore(db_path=vector_store_path, collection_name=collection_name)
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)

        # Define a basic prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["resume_text", "query"],
            template="You are an assistant that reviews resumes.\n"
                     "Given the resume:\n{resume_text}\n\n"
                     "Answer the user's query: {query}"
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def search_resumes(self, query: str, top_k: int = 3):
        """
        Search resumes in vector store for the most relevant ones.
        """
        results = self.vstore.query(query, top_k=top_k)
        return results

    def answer_query(self, query: str, top_k: int = 3):
        """
        Get relevant resumes and generate an answer using LLM.
        """
        results = self.search_resumes(query, top_k=top_k)
        responses = []

        for i, resume_text in enumerate(results['documents'][0]):
            answer = self.chain.run(resume_text=resume_text, query=query)
            responses.append({
                "resume_index": i,
                "resume_text": resume_text,
                "answer": answer
            })

        return responses


# ---------------- Test Run ----------------
if __name__ == "__main__":
    agent = ResumeAgent(vector_store_path="./vector_store")

    user_query = "Find candidates with Python and Data Science experience"
    responses = agent.answer_query(user_query, top_k=3)

    for res in responses:
        print(f"\n----- Resume {res['resume_index']} -----")
        print(f"Answer:\n{res['answer']}\n")
