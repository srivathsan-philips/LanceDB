from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import lancedb
import requests
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

app = FastAPI()

# Initialize LanceDB
db = lancedb.connect("data/sample-lancedb")

class Document(BaseModel):
    text: str

class SearchQuery(BaseModel):
    query: str
    db_name: str

class IndexRequest(BaseModel):
    file_path: str
    table_name: str

# Load your model
model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

class LanceVectorDB(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()


@app.get("/lancedb_test/v1/healthcheck")
def healthcheck() -> str:
    """
    Health check endpoint.
    :return: A string indicating the health status & Display Health Status.
    """
    return '{"Health": "OK"}'

@app.post("/lancedb_test/v1/index")
async def index_documents(request: IndexRequest):
    
    documents = get_unstructured_output(file_path=request.file_path)
    if request.table_name in db.table_names():
        table = db.open_table(request.table_name)
    else:
        table = db.create_table(request.table_name, schema=LanceVectorDB)
    
    print("Indexing documents")
    table.add([{'text':doc['text']} for doc in documents])
    return {"message": "Documents indexed successfully"}

@app.post("/lancedb_test/v1/search")
async def search_documents(query: SearchQuery):
    table = db.open_table(query.db_name)
    # print("Opened table")
    results = table.search(query.query).to_list()
    # print("Results",results)
    # final_results = results.to_dict(orient='records')
    # print("Json serialized",final_results)
    return results


def get_unstructured_output(file_path = '/Users/srivathsan/Documents/GitHub/LanceDB/state_of_the_union.txt'):
    url = "https://www.itaap.philips.com/v1/idp/parsedoc"

    payload = {'chunking_strategy': 'title'}
    files=[
    ('files',('state_of_the_union.txt',open(file_path,'rb'),'text/plain'))
    ]
    headers = {
        'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Ikg5bmo1QU9Tc3dNcGhnMVNGeDdqYVYtbEI5dyIsImtpZCI6Ikg5bmo1QU9Tc3dNcGhnMVNGeDdqYVYtbEI5dyJ9.eyJhdWQiOiJodHRwczovL2NvZ25pdGl2ZXNlcnZpY2VzLmF6dXJlLmNvbSIsImlzcyI6Imh0dHBzOi8vc3RzLndpbmRvd3MubmV0LzFhNDA3YTJkLTc2NzUtNGQxNy04NjkyLWIzYWMyODUzMDZlNC8iLCJpYXQiOjE3MjcyNjE5MDcsIm5iZiI6MTcyNzI2MTkwNywiZXhwIjoxNzI3MjY1ODA3LCJhaW8iOiJFMmRnWUhEMldxTFZYaFRxRlpSdno3UnUvZmNiQUE9PSIsImFwcGlkIjoiMmNlOTVmYjItZWJlMC00Mjg0LWJhMzItM2U1NjFmZTQ1YzM2IiwiYXBwaWRhY3IiOiIxIiwiaWRwIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvMWE0MDdhMmQtNzY3NS00ZDE3LTg2OTItYjNhYzI4NTMwNmU0LyIsImlkdHlwIjoiYXBwIiwib2lkIjoiMjJiNzMwZWMtNGQzOS00ZGU2LTlkYTYtZjY1OTEwNmNkN2Q3IiwicmgiOiIwLkFRa0FMWHBBR25WMkYwMkdrck9zS0ZNRzVKQWlNWDNJS0R4SG9PMk9VM1NiYlcwSkFBQS4iLCJzdWIiOiIyMmI3MzBlYy00ZDM5LTRkZTYtOWRhNi1mNjU5MTA2Y2Q3ZDciLCJ0aWQiOiIxYTQwN2EyZC03Njc1LTRkMTctODY5Mi1iM2FjMjg1MzA2ZTQiLCJ1dGkiOiJCUlBnRGV1WkswcXktN2E2akdNTUFBIiwidmVyIjoiMS4wIiwieG1zX2lkcmVsIjoiNyAxMCJ9.bk8iW28F6miivl3PYPlXWHK7SM-MpZBx7ell1xoAPh7Gl-keECy0aT4YZlbD7xCYS4r4wg4dkMP6faoEbbT4oOtOaNOhYyuVtWX8IS8LfrsgUF-iQp7RIFSne4GTnXe3WKNwGkvXFiU4vKofA5bs33Ld2AtgBHRSjF87QnbZVrm7D8MmQaZ9362c-W5rxQK7_d8qapA6k0kz3nIAtL8gCoYfwobGdFHVFJFBdjWEY_TNS_bS3PjosFmasuQWKuZEh7g66tduZNvgBf_pOOUCXCp4_O-oaS_DRA1Y5lMjuKTQg4clpPHFUjMlWISRdIFVp-vhdgkUPmxDVdcMnliejg'
    } 
    print("Requesting to parse the document")
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return response.json()
