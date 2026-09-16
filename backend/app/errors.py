class ModelGuardError(Exception):
    def __init__(self, message: str, code: str = "invalid_request", status_code: int = 400, details=None):
        super().__init__(message)
        self.message, self.code, self.status_code, self.details = message, code, status_code, details


def require(value, message: str, code: str = "not_found"):
    if not value:
        raise ModelGuardError(message, code, 404)
    return value
