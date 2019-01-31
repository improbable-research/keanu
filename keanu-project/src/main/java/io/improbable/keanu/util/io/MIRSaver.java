package io.improbable.keanu.util.io;

import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.vertices.Vertex;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;

public class MIRSaver implements NetworkSaver {
    @Override
    public void save(OutputStream output, boolean saveValues, Map<String, String> metadata) throws IOException {

    }

    @Override
    public void save(Vertex vertex) {

    }

    @Override
    public void saveValue(Vertex vertex) {

    }
}
