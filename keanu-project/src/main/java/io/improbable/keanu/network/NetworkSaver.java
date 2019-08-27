package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;

public interface NetworkSaver {

    void save(OutputStream output, boolean saveValues, Map<String, String> metadata) throws IOException;

    default void save(OutputStream output, boolean saveValues) throws IOException {
        save(output, saveValues, null);
    }

    void save(Vertex vertex);

    void saveValue(Vertex vertex);
}
