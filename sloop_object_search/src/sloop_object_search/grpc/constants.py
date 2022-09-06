import re

class Message:
    """message used as communication between server and client.
    A message is """
    REQUEST_LOCAL_SEARCH_REGION_UPDATE = "Request local search region update [for {}]"

    @staticmethod
    def match(m):
        if m.startswith("Request local search region update"):
            return Message.REQUEST_LOCAL_SEARCH_REGION_UPDATE

    @staticmethod
    def forwhom(m):
        match = re.search("\[for [^\[\]]*\]", m)
        if match:
            return match.group(0)[5:-1].strip()

class Info:
    LOCAL_SEARCH_REGION = "local search region [for {}]"
