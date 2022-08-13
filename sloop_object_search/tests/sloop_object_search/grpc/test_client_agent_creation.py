import grpc

from sloop_object_search.grpc.client import SloopObjectSearchClient

def run():
    client = SloopObjectSearchClient()
    response = client.CreateAgent(
        config_file_path="./config_file_test_topo2d.yaml")
    print("Sloop Object Search client received: " + response.message)


if __name__ == "__main__":
    run()
