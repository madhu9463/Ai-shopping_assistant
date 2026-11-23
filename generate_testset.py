
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.documents import Document

try:
    from ragas.testset import TestsetGenerator
except ImportError as e:
    print(
        f"RAGAS Import Error: {e}. "
        "Please make sure you have installed all "
        "the required libraries."
    )
    exit(1)


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print(
            "Error: Please add GEMINI_API_KEY "
            "to your .env file."
        )
        return

    os.environ["GOOGLE_API_KEY"] = api_key

    # ==========================================================
    # Step 1: Load Product Dataset
    # ==========================================================

    print("1. Loading product data from data/rag_dataset.csv...")

    try:
        from build_index import (
            load_dataset,
            create_langchain_documents,
        )

        df = load_dataset("data/rag_dataset.csv")
        raw_documents = create_langchain_documents(df)

    except Exception as e:
        print(f"Error loading product dataset: {e}")
        return

    # Combine all products into one large text document
    # to prevent errors caused by documents being too short.
    #
    # RAGAS 0.2.x requires documents longer than 100 tokens
    # for knowledge graph analysis.

    combined_text = "\n\n---\n\n".join(
        [doc.page_content for doc in raw_documents]
    )

    documents = [
        Document(page_content=combined_text)
    ]

    # ==========================================================
    # Step 2: Initialize Testset Generator
    # ==========================================================

    print("2. Initializing AI testset generator (Gemini)...")

    # Initialize Google language and embedding models.
    # Gemini 2.5 Flash is used for generating test questions.

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # Initialize TestsetGenerator (RAGAS 0.2.x)

    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings,
    )

    # ==========================================================
    # Step 3: Generate Evaluation Questions
    # ==========================================================

    print(
        "3. Generating 15 evaluation questions "
        "automatically using RAGAS..."
    )

    try:
        from ragas.run_config import RunConfig

        # Limit execution to one worker.
        # Configure retries to handle temporary API errors
        # and rate limits.

        run_config = RunConfig(
            max_workers=1,
            max_retries=100,
        )

        # Generate test questions from the knowledge graph.
        # Compatible with the original RAGAS 0.2.x workflow.

        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=15,
            run_config=run_config,
        )

    except Exception as e:
        print(f"\n[!] Error generating testset: {e}")

        print(
            "Suggestion: This may be caused by Gemini API "
            "rate limits. Try reducing testset_size to 5."
        )
        return

    # ==========================================================
    # Step 4: Convert and Save Generated Testset
    # ==========================================================

    print("4. Converting and saving generated results...")

    # Convert generated testset into a Pandas DataFrame.
    # Save the DataFrame as a CSV file.

    df = testset.to_pandas()

    output_path = "data/ragas_testset.csv"

    df.to_csv(output_path, index=False)

    print("\nTestset generation completed successfully!")

    print(
        f"Generated testset saved to: {output_path}"
    )

    print(
        "You can open the CSV file to view "
        "the AI-generated evaluation questions."
    )


if __name__ == "__main__":
    main()
