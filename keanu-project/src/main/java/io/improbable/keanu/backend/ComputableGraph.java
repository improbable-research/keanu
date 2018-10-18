package io.improbable.keanu.backend;

import java.util.Collection;
import java.util.Map;

public interface ComputableGraph extends AutoCloseable {

    <T> T eval(Map<String, ?> inputs, String output);

    Map<String, ?> eval(Map<String, ?> inputs, Collection<String> outputs);

    @Override
    void close();
}
