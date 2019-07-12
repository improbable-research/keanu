package io.improbable.keanu.util.io;

import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.mir.KeanuSavedBayesNet;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Map;

/**
 * A class for outputting a network to a JSON file.
 * JSON output file contains information about the network (vertices, their types and connections),
 * network state (values for constant and observed vertices, if saveValues parameter is set to true),
 * as well as metadata (string key - value pairs, if metadata is passed in).
 * <p>
 * Usage:
 * Create jsonSaver: JsonSaver jsonSaver = new JsonSaver(yourBayesianNetwork);
 * or JsonSaver jsonSaver = new JsonSaver(yourBayesianNetwork, metadata);
 * where metadata is a map between some keys and values in string format (for instance, "author": "Jane Doe").
 * To output network to a JSON file: jsonSaver.save(outputStream, saveValues);
 */
public class JsonSaver implements NetworkSaver {

    private final ProtobufSaver protobufSaver;

    /**
     * Sets up a new json saver for the given network.
     *
     * @param net network that will be saved
     */
    public JsonSaver(BayesianNetwork net) {
        protobufSaver = new ProtobufSaver(net);
    }

    @Override
    public void save(OutputStream output, boolean saveValues, Map<String, String> metadata) throws IOException {
        KeanuSavedBayesNet.ProtoModel model = protobufSaver.getModel(saveValues, metadata);

        Writer outputWriter = new OutputStreamWriter(output);
        String jsonOutput = JsonFormat.printer().print(model);
        outputWriter.write(jsonOutput);
        outputWriter.close();
    }

    @Override
    public void save(Vertex vertex) {
        protobufSaver.save(vertex);
    }

    @Override
    public void saveValue(Vertex vertex) {
        protobufSaver.save(vertex);
    }

}
