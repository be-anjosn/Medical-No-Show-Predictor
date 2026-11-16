from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class AppointmentInput(BaseModel):
    gender: Literal["F", "M"]
    age: int = Field(ge=0, le=115, description="Pacient Age", examples=[35])
    scholarship: int = Field(ge=0, le=1, description="Bolsa Família (0 or 1)", examples=[0])
    hipertension: int = Field(ge=0, le=1, examples=[0])
    diabetes: int = Field(ge=0, le=1, examples=[0])
    alcoholism: int = Field(ge=0, le=1, examples=[0])
    handcap: int = Field(ge=0, le=4, examples=[0])
    scheduled_day: datetime = Field(description="Date that the appointment was made", examples=["2026-06-01T10:00:00"])
    appointment_day: datetime = Field(description="Date of the appointment", examples=["2026-06-10T14:30:00"])

    @field_validator("appointment_day")
    @classmethod
    def validate_appointment_day(cls, value: datetime, info) -> datetime:
        scheduled_day = info.data.get("scheduled_day")

        if scheduled_day and value.date() < scheduled_day.date():
            raise ValueError(
                "Appointment date cannot be before scheduled date"
            )
        return value

    @field_validator('age')
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < 0:
            raise ValueError('Age cannot be negative')

        return value


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    prediction_label: Literal["attended", "possible_no_show"]
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


class ExplanationResponse(BaseModel):
    explanation: str = Field(min_length=1)
    model: str


class AnalysisResponse(BaseModel):
    prediction: PredictionResponse
    explanation: ExplanationResponse