from pydantic_settings import BaseSettings, SettingsConfigDict
import httpx

class Config (BaseSettings):

    AUTHORIZATION_KEY : str
    ACCESS_TOKEN : str|None = None
    
    model_config = SettingsConfigDict(
            env_file=".env",
            extra="ignore"
        )

    def update_access_token(self):
        if self.AUTHORIZATION_KEY:
            url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

            payload = {
                'scope': 'GIGACHAT_API_PERS'
            }
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'RqUID': '764c550a-704c-48fd-964e-b6038fad5b38',
                'Authorization': f'Basic {self.AUTHORIZATION_KEY}'
            }

            with httpx.Client(verify=False) as client:
                response = client.post(url, headers=headers, data=payload)

                if response.status_code == 200:
                    self.ACCESS_TOKEN = response.json()['access_token']


config = Config()