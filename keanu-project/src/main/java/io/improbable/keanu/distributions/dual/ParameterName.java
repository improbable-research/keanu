package io.improbable.keanu.distributions.dual;

import java.util.Objects;

public class ParameterName {
    public static final ParameterName MIN = new ParameterName("MIN");
    public static final ParameterName MAX = new ParameterName("MAX");
    public static final ParameterName A = new ParameterName("A");
    public static final ParameterName B = new ParameterName("B");
    public static final ParameterName K = new ParameterName("K");
    public static final ParameterName N = new ParameterName("N");
    public static final ParameterName P = new ParameterName("P");
    public static final ParameterName S = new ParameterName("S");
    public static final ParameterName T = new ParameterName("T");
    public static final ParameterName V = new ParameterName("V");
    public static final ParameterName X = new ParameterName("X");
    public static final ParameterName BETA = new ParameterName("BETA");
    public static final ParameterName MU = new ParameterName("MU");
    public static final ParameterName SIGMA = new ParameterName("SIGMA");
    public static final ParameterName THETA = new ParameterName("THETA");

    private final String name;

    public ParameterName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParameterName parameterName = (ParameterName) o;
        return Objects.equals(name, parameterName.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    @Override
    public String toString() {
        return getName();
    }
}
