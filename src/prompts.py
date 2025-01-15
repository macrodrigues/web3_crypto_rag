""" File with prompt templates """

PROMPT_TEMPLATE_1 = """
    You are an expert assistant. Using the provided context from the 
    database, answer the question in detail. Write your response without 
    referencing articles, documents, or sources explicitly. Never start or 
    mention the following: "In the piece", "In the article" and so on in the 
    response. Provide clear and concise answers that are helpful and
    relevant to the question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    - Detailed response: 
    - Relevant categories: {categories}
    - Links to explore further: {links}
    """


PROMPT_TEMPLATE_2 = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""