package io.improbable.keanu.vertices;

import io.improbable.keanu.network.NetworkWriter;

public interface SaveableVertex {

    void save(NetworkWriter netWriter);
    void saveValue(NetworkWriter netWriter);
}
