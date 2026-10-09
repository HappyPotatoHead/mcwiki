class Model:
    def __init__(self, model_name: str, device: str | None = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
