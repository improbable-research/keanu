package io.improbable.keanu.distributions.dual;

import java.util.Map;
import java.util.NoSuchElementException;

import com.google.common.collect.Maps;

public class ParameterMap<T> {
    private final Map<ParameterName, ParameterValue<T>> entries = Maps.newHashMap();

    public ParameterMap<T> put(ParameterName id, T value) {
        if (entries.keySet().contains(id)) {
            throw new IllegalArgumentException("Parameter " + id + " has already been set");
        }
        entries.put(id, new ParameterValue(id, value));
        return this;
    }

    public ParameterValue<T> get(ParameterName id) {
        ParameterValue<T> diff = entries.get(id);
        if (diff == null) {
            throw new NoSuchElementException("Cannot find entry for parameter " + id);
        }
        return diff;
    }

    @Override
    public String toString() {
        return entries.toString();
    }

    public int size() {
        return entries.size();
    }
}
