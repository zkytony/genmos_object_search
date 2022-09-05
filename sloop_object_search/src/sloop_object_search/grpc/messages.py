import re

class Message:
    """message used as communication between server and client.
    A message is """
    REQUEST_SEARCH_REGION_UPDATE = "Request UpdateSearch Region [for {}]"

    @staticmethod
    def match(m):
        if m.startswith("Request UpdateSearch Region"):
            return Message.REQUEST_SEARCH_REGION_UPDATE

    @staticmethod
    def forwhom(m):
        match = re.search("\[for [^\[\]]*\]", m)
        if match:
            return match.group(0)[5:-1].strip()
