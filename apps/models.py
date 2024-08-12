from pydantic import BaseModel, Field

class AppFeatures(BaseModel):
    Android_Ver: str = Field(..., example="10")
    Size: float = Field(..., example=100.0)
    Price: float = Field(..., example=0.0)
    Category_encoded: int = Field(..., example=1)
    Type_Free: int = Field(..., example=1)
    Type_Paid: int = Field(..., example=0)
    Content_Ratings_Adults_only_18: int = Field(..., example=0)
    Content_Ratings_Everyone: int = Field(..., example=1)
    Content_Ratings_Everyone_10: int = Field(..., example=0)
    Content_Ratings_Mature_17: int = Field(..., example=0)
    Content_Ratings_Teen: int = Field(..., example=1)
    Content_Ratings_Unrated: int = Field(..., example=0)
    last_updated_year: int = Field(..., example=2024)
    last_updated_month: int = Field(..., example=8)
