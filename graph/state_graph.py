from typing import TypedDict
from langchain_core.documents import Document
import operator
from typing import Annotated
class GraphState(TypedDict):
    question: Annotated[str, operator.add]
    documents: Annotated[list[Document], operator.add]
    web_search: str
    generation: str
