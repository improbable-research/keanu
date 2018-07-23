package io.improbable.keanu.distributions.dual;

import java.util.NoSuchElementException;
import java.util.TreeSet;

import com.google.common.collect.Sets;

public class ParameterMap<T> {
    private final TreeSet<ParameterValue<T>> entries = Sets.newTreeSet();

    public ParameterMap<T> put(ParameterName id, T value) {
        ParameterValue<T> entry = new ParameterValue(id, value);
        if (entries.contains(entry)) {
            throw new IllegalArgumentException("Parameter " + id + " has already been set");
        }
        entries.add(entry);
        return this;
    }

    public ParameterValue<T> get(ParameterName id) {
        ParameterValue<T> diff = entries.floor(new ParameterValue<T>(id));
        if (diff == null) {
            throw new NoSuchElementException("Cannot find entry for parameter " + id);
        }
        return diff;
    }
}
