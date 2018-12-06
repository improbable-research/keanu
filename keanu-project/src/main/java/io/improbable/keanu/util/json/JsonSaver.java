package io.improbable.keanu.util.json;

import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.ProtobufSaver;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Map;

/**
 * Utility class for outputting a network to a JSON file.
 * JSON output file contains information about the network (vertices, their types and connections),
 * network state (values for constant and observed vertices, if saveValues parameter is set to true),
 * as well as metadata (string key - value pairs, if metadata is passed in).
 *
 * Usage:
 * Create jsonSaver: JsonSaver jsonSaver = new JsonSaver(yourBayesianNetwork);
 * or JsonSaver jsonSaver = new JsonSaver(yourBayesianNetwork, metadata);
 * where metadata is a map between some keys and values in string format (for instance, "author": "Jane Doe").
 * To output network to a JSON file: jsonSaver.save(outputStream, saveValues);
 */
public class JsonSaver extends ProtobufSaver {

    private final Map<String, String> metadata;

    /**
     * Sets up a new json saver for the given network.
     * @param net network that will be saved
     */
    public JsonSaver(BayesianNetwork net) {
        this(net, null);
    }

    /**
     * Sets up a new json saver for the given network.
     * @param net network that will be saved
     * @param metadata metadata to add to the json output file
     */
    public JsonSaver(BayesianNetwork net, Map<String, String> metadata) {
        super(net);
        this.metadata = metadata;
    }

    @Override
    public void save(OutputStream output, boolean saveValues) throws IOException {
        modelBuilder = KeanuSavedBayesNet.Model.newBuilder();

        saveMetadata();
        net.save(this);

        if (saveValues) {
            net.saveValues(this);
        }
        Writer outputWriter = new OutputStreamWriter(output);
        String jsonOutput = JsonFormat.printer().print(modelBuilder);
        outputWriter.write(jsonOutput);
        outputWriter.close();
        modelBuilder = null;
    }

    private void saveMetadata() {
        if (metadata != null) {
            KeanuSavedBayesNet.Metadata.Builder metadataBuilder = KeanuSavedBayesNet.Metadata.newBuilder();
            for (Map.Entry<String, String> entry : metadata.entrySet()) {
                metadataBuilder.putMetadataInfo(entry.getKey(), entry.getValue());
            }
            modelBuilder.setMetadata(metadataBuilder);
        }
    }
}
