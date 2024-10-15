import base64
from django import template

register = template.Library()

@register.filter(name='b64encode')
def b64encode(value):
    # Chuyển đổi giá trị sang base64
    return base64.b64encode(value).decode('utf-8')
