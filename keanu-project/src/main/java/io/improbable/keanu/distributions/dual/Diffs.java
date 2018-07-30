package io.improbable.keanu.distributions.dual;

import java.util.NoSuchElementException;
import java.util.TreeSet;

import com.google.common.collect.Sets;

import io.improbable.keanu.tensor.number.dbl.DoubleTensor;

public class Diffs {
    public static final ParameterName A = new ParameterName("A");
    public static final ParameterName B = new ParameterName("B");
    public static final ParameterName K = new ParameterName("K");
    public static final ParameterName S = new ParameterName("S");
    public static final ParameterName T = new ParameterName("T");
    public static final ParameterName X = new ParameterName("X");
    public static final ParameterName BETA = new ParameterName("BETA");
    public static final ParameterName MU = new ParameterName("MU");
    public static final ParameterName SIGMA = new ParameterName("SIGMA");
    public static final ParameterName THETA = new ParameterName("THETA");

    private final TreeSet<Diff> diffs = Sets.newTreeSet();

    public Diffs put(ParameterName id, DoubleTensor value) {
        Diff diff = new Diff(id, value);
        if (diffs.contains(diff)) {
            throw new IllegalArgumentException("Diff named " + id + " has already been set");
        }
        diffs.add(diff);
        return this;
    }

    public Diff get(ParameterName id) {
        Diff diff = diffs.floor(new Diff(id));
        if (diff == null) {
            throw new NoSuchElementException("Cannot find Diff named " + id);
        }
        return diff;
    }
}
