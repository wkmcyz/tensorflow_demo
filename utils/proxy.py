import os

def set_proxy():
    PROXY = os.environ['VPN_PROXY']
    os.environ["http_proxy"] = PROXY
    os.environ["https_proxy"] = PROXY
