"""用户画像模块异常。"""

from app.core.exceptions import ApplicationError


class ProfileFieldInvalid(ApplicationError):
    """画像字段名不在白名单内。"""

    status_code: int = 400
    code: str = "PROFILE_FIELD_INVALID"
    message: str = "不支持的画像字段"


class ProfileValueInvalid(ApplicationError):
    """画像字段值格式不合法。"""

    status_code: int = 400
    code: str = "PROFILE_VALUE_INVALID"
    message: str = "画像字段值无效"
