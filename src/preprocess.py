# pylint: disable=E0401
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class GoogleAccess:
    """ Class containing the Google interaction functions """

    @staticmethod
    def google_authentication(credentials) -> object:
        """ This function gives authentication to the Google Account.

        Scopes defines the permissions to Googe Sheets and Google Drive.
        Than, using gspread, it authorizes the authentication and
        opens the the worksheet to work with.

        """
        # Authenticate with Google Sheets using the JSON key file
        scope = ['https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive']

        creds = Credentials.from_service_account_file(
            credentials, scopes=scope)

        client = gspread.authorize(creds)

        return client

    @staticmethod
    def read_from_sheet(client, sheet_id) -> pd.DataFrame:
        """ This function reads the data from the Google Sheet. """

        # Open the Google Sheet by id
        sheet = client.open_by_key(sheet_id)

        ws = sheet.worksheet('main')
        df = pd.DataFrame(data=ws.get_all_records())
        return df

def create_documents(credentials_google, google_sheet_id):
    """ This function splits the articles in paragraphas (chunks) 
    from a dataset obtained by reading a google sheet. 
    And creates documents with the chunk, an id, and a category.
    """

    # Create client
    client = GoogleAccess.google_authentication(credentials_google)
    df = GoogleAccess.read_from_sheet(client, google_sheet_id)
    df['id'] = df['id'].astype('str')

    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=0)

    # Create a list of LangChain documents
    documents = []

    for _, row in df.iterrows():
        content_chunks = splitter.split_text(row['content'])
        for i, chunk in enumerate(content_chunks):
            # Generate a unique UUID4 for each chunk
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "category": row['category'],
                    "link": row['link']
                },
                id=f"{row['id']}{i}"
            ))

    return documents
