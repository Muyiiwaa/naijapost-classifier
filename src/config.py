from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    hugging_face_api_key: str 
    model_name: str 
    num_labels: int
    categories: list[str] 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


if __name__ == "__main__":
    settings = Settings()
    print(settings.model_name)
    print(settings.num_labels)