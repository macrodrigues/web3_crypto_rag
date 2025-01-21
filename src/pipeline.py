from langchain_core.prompts import PromptTemplate
import os
from langchain_ollama.llms import OllamaLLM
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("keys.env")


def query_rag(query_text, db, prompt):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database,
    and format the response with links and categories.
    """

    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    print(results)
    print(results[0][1])

    # Check if there are any matching results or if
    # the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.15:
        return (
            "No relevant information found. Try to ask something "
            "related to crypto, blockchain and web3.")

    else:
        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join(
            [doc.page_content for doc, _score in results]
        )

        # Extract categories and links from metadata
        categories = ", ".join(
            {doc.metadata.get("category", "Unknown") for doc, _ in results})
        links = ", ".join(
            {doc.metadata.get("link", "N/A") for doc, _ in results})

        # Create and format the prompt
        template = PromptTemplate(
            input_variables=["context", "question", "categories", "links"],
            template=prompt,
        )
        prompt = template.format(
            context=context_text,
            question=query_text,
            categories=categories,
            links=links
        )

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("API_KEY"),
            )

        completion = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ]
        )

        response_text = completion.choices[0].message.content

        # # Use the language model to generate a response
        # model = OllamaLLM(
        #     model=model_name, cache=False, verbose=True, num_ctx=500)
        # response_text = model.invoke(prompt)

        # Format and return response including generated text and sources
        formatted_response = f"{response_text}\nSources: {links}"

        return formatted_response
