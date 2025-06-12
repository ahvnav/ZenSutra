# app/schemas.py
from pydantic import BaseModel
from typing import Literal

class CustomerFeatures(BaseModel):
    age: int
    gender: Literal['M', 'F'] # Ensures input is either 'M' or 'F'
    monthly_bill: float
    data_usage_gb: float
    contract_type: Literal['Monthly', 'Annual', 'Two Year'] # Matches your mock data contract types

    class Config:
        schema_extra = { # type: ignore
            "example": {
                "age": 30,
                "gender": "M",
                "monthly_bill": 75.50,
                "data_usage_gb": 22.1,
                "contract_type": "Annual"
            }
        }