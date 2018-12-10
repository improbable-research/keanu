package io.improbable.keanu.util.json;

import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.ProtobufLoader;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class JsonLoader implements NetworkLoader {

    ProtobufLoader protobufLoader = new ProtobufLoader();

    @Override
    public BayesianNetwork loadNetwork(InputStream input) throws IOException {
        String jsonInput = IOUtils.toString(input, StandardCharsets.UTF_8);
        KeanuSavedBayesNet.Model.Builder modelBuilder = KeanuSavedBayesNet.Model.newBuilder();
        JsonFormat.parser().merge(jsonInput, modelBuilder);
        return protobufLoader.loadNetwork(modelBuilder.build());
    }

    @Override
    public void loadValue(DoubleVertex vertex) {
        protobufLoader.loadValue(vertex);
    }

    @Override
    public void loadValue(BoolVertex vertex) {
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
