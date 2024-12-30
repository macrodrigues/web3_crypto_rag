""" File with prompt templates """

PROMPT_TEMPLATE_1 = """
    You are an expert assistant. Using the provided context from the 
    database, answer the question in detail. Include relevant categories 
    and links as part of your response if applicable.

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