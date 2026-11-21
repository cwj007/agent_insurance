"""应用异常基类"""

class ApplicationError(Exception):
    """应用异常基类"""

    status_code: int = 500  # 默认状态码
    code: str = "APPLICATION_ERROR"  # 异常编码
    message: str = "服务器内部异常错误"  # 异常消息