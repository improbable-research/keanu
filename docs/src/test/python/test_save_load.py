from keanu.vertex import Gamma
from keanu import BayesNet
from keanu.network_io import ProtobufLoader, JsonLoader, ProtobufSaver, DotSaver, JsonSaver

def test_can_save_and_load(tmpdir) -> None:
    PROTO_FILE = str(tmpdir.join("test.proto"))
    JSON_FILE = str(tmpdir.join("test.json"))
    DOT_FILE = str(tmpdir.join("test.dot"))

    gamma = Gamma(1.0, 1.0)
    gamma.set_value(2.5)
    net = BayesNet(gamma.get_connected_graph())
    metadata = {"Team": "GraphOS"}
    protobuf_saver = ProtobufSaver(net)
    protobuf_saver.save(PROTO_FILE, True, metadata)
    json_saver = JsonSaver(net)
    json_saver.save(JSON_FILE, True, metadata)
    dot_saver = DotSaver(net)
    dot_saver.save(DOT_FILE, True, metadata)
    check_dot_file(DOT_FILE)

    protobuf_loader = ProtobufLoader()
    json_loader = JsonLoader()
    new_net_from_proto = protobuf_loader.load(PROTO_FILE)
    check_loaded_net(new_net_from_proto)
    new_net_from_json = json_loader.load(JSON_FILE)
    check_loaded_net(new_net_from_json)