package io.improbable.keanu.network;

import java.io.IOException;
import java.io.InputStream;

public interface NetworkReader {

    public BayesianNetwork loadNetwork(InputStream input) throws IOException;
}
