from pydantic import BaseModel

class Diamond(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

class DiamondResponse(Diamond):
    price: float

class Diamonds(BaseModel):
    diamonds: list[Diamond]