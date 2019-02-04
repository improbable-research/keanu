package io.improbable.keanu.util.io;

import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.mir.KeanuSavedBayesNet;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * A class for parsing a network from JSON format to a BayesianNetwork object.
 *
 * Usage:
 * Create jsonLoader: JsonLoader jsonLoader = new JsonLoader();
 * Loading in a network: BayesianNetwork net = jsonLoader.loadNetwork(inputStream);
 * where inputStream is a stream of network in JSON format.
 */
public class JsonLoader implements NetworkLoader {

    private final ProtobufLoader protobufLoader = new ProtobufLoader();

    @Override
    public BayesianNetwork loadNetwork(InputStream input) throws IOException {
        String jsonInput = IOUtils.toString(input, StandardCharsets.UTF_8);
        KeanuSavedBayesNet.ProtoModel.Builder modelBuilder = KeanuSavedBayesNet.ProtoModel.newBuilder();
        JsonFormat.parser().merge(jsonInput, modelBuilder);
        return protobufLoader.loadNetwork(modelBuilder.build());
    }

    @Override
    public void loadValue(DoubleVertex vertex) {
        protobufLoader.loadValue(vertex);
    }

    @Override
    public void loadValue(BooleanVertex vertex) {
        protobufLoader.loadValue(vertex);
    }

    @Override
    public void loadValue(IntegerVertex vertex) {
        protobufLoader.loadValue(vertex);
    }

    @Override
    public void loadValue(Vertex vertex) {
        protobufLoader.loadValue(vertex);
    }
}
