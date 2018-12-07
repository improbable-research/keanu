package io.improbable.keanu.util.json;

import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.ProtobufLoader;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class JsonLoader extends ProtobufLoader {

    @Override
    public BayesianNetwork loadNetwork(InputStream input) throws IOException {
        String jsonInput = IOUtils.toString(input, StandardCharsets.UTF_8);
        KeanuSavedBayesNet.Model.Builder modelBuilder = KeanuSavedBayesNet.Model.newBuilder();
        JsonFormat.parser().merge(jsonInput, modelBuilder);
        return super.loadNetwork(modelBuilder.build());
    }
}
