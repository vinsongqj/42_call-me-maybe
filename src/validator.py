from pydantic import BaseModel
from typing import Dict, Any


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]