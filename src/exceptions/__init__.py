


class ConfigurationException(Exception):
    def __init__(self, key, msg) -> None:

        self.key = key 
        self.msg = msg 
        super().__init__()
    
    def __str__(self) -> str:
        return f"{self.key}: {self.msg}" 