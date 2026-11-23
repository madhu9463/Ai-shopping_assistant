
import pandas as pd
from datasets import Dataset
import os
from dotenv import load_dotenv

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install all dependencies from requirements.txt")
    exit(1)


from rag_pipeline import load_rag_chain, ask


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: Please add GEMINI_API_KEY to your .env file.")
        return

    # Set the Google API key for LangChain
    os.environ["GOOGLE_API_KEY"] = api_key

    # ==========================================================
    # Step 1: Load the RAG System
    # ==========================================================
    # Previously: Loaded 5 components separately:
    # embedding_model, faiss_index, docs, bm25, reranker
    #
    # Now: load_rag_chain() loads the complete RAG pipeline.

    print("1. Loading the RAG system (LangChain Chain)...")

    try:
        chain = load_rag_chain()
    except Exception as e:
        print(f"Error loading RAG system: {e}")
        return

    # ==========================================================
    # Step 2: Load the Evaluation Test Dataset
    # ==========================================================

    # ==========================================================
# Step 2: Load the Evaluation Test Dataset
# ==========================================================

    print("2. Loading evaluation test dataset...")

    try:
        testset_df = pd.read_csv("data/ragas_testset_english.csv")
    except FileNotFoundError:
        print("Error: data/ragas_testset_english.csv not found.")
        return

    questions = testset_df["question"].tolist()

    if "ground_truth" in testset_df.columns:
        ground_truths = testset_df["ground_truth"].tolist()
    else:
        print("Warning: ground_truth column not found.")
        ground_truths = [""] * len(questions)

    print(f"Total test cases: {len(questions)}")

    # print("2. Loading evaluation test dataset...")

    # try:
    #     testset_df = pd.read_csv("data/ragas_testset.csv")
    # except FileNotFoundError:
    #     print(
    #         "Error: data/ragas_testset.csv was not found. "
    #         "Please run generate_testset.py first."
    #     )
    #     return

    # questions = testset_df["question"].tolist()

    # # Check whether the dataset contains ground truth answers
    # if "ground_truth" in testset_df.columns:
    #     ground_truths = testset_df["ground_truth"].tolist()
    # else:
    #     print(
    #         "Warning: The ground_truth column was not found. "
    #         "RAGAS may not be able to calculate context recall."
    #     )
    #     ground_truths = [""] * len(questions)

    # answers = []
    # contexts_list = []

    # ==========================================================
    # Step 3: Collect Answers from the RAG System
    # ==========================================================
    # Previously: Performed multiple manual operations:
    # 1. Encode the user query
    # 2. Retrieve relevant documents
    # 3. Build the prompt
    # 4. Generate the answer
    #
    # Now: ask(chain, question) handles the complete process.

    print("3. Collecting RAG answers for evaluation...")

    for idx, question in enumerate(questions):

        print(
            f"   Processing question {idx + 1}/{len(questions)}..."
        )

        result = ask(chain, question)

        # Extract the generated answer
        answer = result["answer"]

        # Extract retrieved contexts
        # LangChain returns Document objects.
        # RAGAS requires a list of strings.

        contexts = [
            doc.page_content for doc in result["context"]
        ]

        answers.append(answer)
        contexts_list.append(contexts)

    # ==========================================================
    # Step 4: Prepare the RAGAS Evaluation Dataset
    # ==========================================================

    print("4. Preparing the dataset for RAGAS evaluation...")

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(data)

    # ==========================================================
    # Step 5: Evaluate the RAG System Using RAGAS
    # ==========================================================

    print("5. Starting RAGAS evaluation using Gemini...")

    # Configure the evaluation LLM
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )

    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # Run RAGAS evaluation
    try:
        from ragas.run_config import RunConfig

        # Evaluate questions with a single worker
        # and retry if temporary API errors occur.

        run_config = RunConfig(
            max_workers=1,
            max_retries=100,
        )

        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
        )

    except Exception as e:
        print(f"\nEvaluation error: {e}")
        return

    # ==========================================================
    # Step 6: Display Evaluation Results
    # ==========================================================

    print("\n================ RAGAS EVALUATION RESULTS ================")
    print(result)
    print("==========================================================")

    # ==========================================================
    # Step 7: Save Evaluation Results to CSV
    # ==========================================================

    result_df = result.to_pandas()

    output_path = "data/evaluation_results.csv"

    result_df.to_csv(output_path, index=False)

    print(
        f"\nEvaluation results successfully saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
