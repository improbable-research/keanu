from keanu.vertex import Gamma
from keanu import BayesNet
from keanu.network_io import ProtobufLoader, JsonLoader, ProtobufSaver, DotSaver, JsonSaver

def test_can_save_and_load(tmpdir) -> None:
    PROTO_FILE_NAME = str(tmpdir.join("test.proto"))
    JSON_FILE_NAME = str(tmpdir.join("test.json"))
    DOT_FILE_NAME = str(tmpdir.join("test.dot"))

    gamma = Gamma(1.0, 1.0)
    gamma.set_value(2.5)
    # %%SNIPPET_START%% PythonSaveSnippet
    net = BayesNet(gamma.get_connected_graph())
    metadata = {"Author": "Documentation Team"}

    protobuf_saver = ProtobufSaver(net)
    protobuf_saver.save(PROTO_FILE_NAME, True, metadata)

    json_saver = JsonSaver(net)
    json_saver.save(JSON_FILE_NAME, True, metadata)

    dot_saver = DotSaver(net)
    dot_saver.save(DOT_FILE_NAME, True, metadata)
    # %%SNIPPET_END%% PythonSaveSnippet

    # %%SNIPPET_START%% PythonLoadSnippet
    protobuf_loader = ProtobufLoader()
    new_net_from_proto = protobuf_loader.load(PROTO_FILE_NAME)
    json_loader = JsonLoader()
    new_net_from_json = json_loader.load(JSON_FILE_NAME)
    # %%SNIPPET_END%% PythonLoadSnippet
