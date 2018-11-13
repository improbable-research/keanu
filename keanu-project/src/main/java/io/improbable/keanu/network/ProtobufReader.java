package io.improbable.keanu.network;

import com.google.common.primitives.Booleans;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.Map;

public class ProtobufReader implements NetworkReader {

    private final Map<Vertex, KeanuSavedBayesNet.VertexValue> savedValues;

    public ProtobufReader() {
        savedValues = new HashMap<>();
    }

    @Override
    public void loadValue(Vertex vertex) {
        throw new IllegalArgumentException("Cannot Load value for Untyped Vertex");
    }

    public BayesianNetwork loadNetwork(InputStream input) throws IOException {
        Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices = new HashMap<>();
        KeanuSavedBayesNet.BayesianNetwork parsedNet = KeanuSavedBayesNet.BayesianNetwork.parseFrom(input);

        for (KeanuSavedBayesNet.Vertex vertex : parsedNet.getVerticesList()) {
            Vertex newVertex = fromProtoBuf(vertex, instantiatedVertices);
            instantiatedVertices.put(vertex.getId(), newVertex);
        }

        BayesianNetwork bayesNet = new BayesianNetwork(instantiatedVertices.values());

        loadDefaultValues(parsedNet, instantiatedVertices, bayesNet);

        return bayesNet;
    }

    @Override
    public DoubleTensor getInitialDoubleValue(Object valueKey) {
        KeanuSavedBayesNet.VertexValue value = (KeanuSavedBayesNet.VertexValue)valueKey;

        return extractDoubleValue(value);
    }

    @Override
    public void loadValue(DoubleVertex vertex) {
        KeanuSavedBayesNet.VertexValue value = savedValues.get(vertex);
        DoubleTensor tensor = extractDoubleValue(value);
        vertex.setValue(tensor);
    }

    private DoubleTensor extractDoubleValue(KeanuSavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.DOUBLEVAL) {
            throw new IllegalArgumentException("Non Double Value specified for Double Vertex");
        } else {
            return DoubleTensor.create(
                Doubles.toArray(value.getDoubleVal().getValuesList()),
                Longs.toArray(value.getDoubleVal().getShapeList()));
        }
    }

    private void loadDefaultValues(KeanuSavedBayesNet.BayesianNetwork parsedNet,
                                          Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                          BayesianNetwork bayesNet) {
        for (KeanuSavedBayesNet.StoredValue value : parsedNet.getDefaultStateList()) {
            Vertex targetVertex = null;

            if (value.hasId()) {
                targetVertex = instantiatedVertices.get(value.getId());
            }

            if (value.getVertexLabel() != "") {
                Vertex newTarget = bayesNet.getVertexByLabel(new VertexLabel(value.getVertexLabel()));

                if (targetVertex != null && newTarget != targetVertex) {
                    throw new IllegalArgumentException(
                        "Label and VertexID don't refer to same Vertex: ("
                            + value.getVertexLabel() + ") ("
                            + value.getId() + ")");
                } else {
                    targetVertex = newTarget;
                }
            }

            if (targetVertex == null) {
                throw new IllegalArgumentException("Value specified for unknown Vertex: ("
                    + value.getVertexLabel() + ") ("
                    + value.getId() + ")");
            }

            savedValues.put(targetVertex, value.getValue());
            targetVertex.loadValue(this);
        }
    }

    @Override
    public void loadValue(BoolVertex vertex) {
        KeanuSavedBayesNet.VertexValue value = savedValues.get(vertex);
        BooleanTensor tensor = extractBoolValue(value);
        vertex.setValue(tensor);
    }

    private BooleanTensor extractBoolValue(KeanuSavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.BOOLVAL) {
            throw new IllegalArgumentException("Non Boolean Value specified for Boolean Vertex");
        } else {
            return BooleanTensor.create(
                Booleans.toArray(value.getBoolVal().getValuesList()),
                Longs.toArray(value.getDoubleVal().getShapeList()));
        }
    }

    @Override
    public void loadValue(IntegerVertex vertex) {
        KeanuSavedBayesNet.VertexValue value = savedValues.get(vertex);
        IntegerTensor tensor = extractIntValue(value);
        vertex.setValue(tensor);
    }

    private IntegerTensor extractIntValue(KeanuSavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.INTVAL) {
            throw new IllegalArgumentException("Non Int Value specified for Int Vertex");
        } else {
            return IntegerTensor.create(
                Ints.toArray(value.getIntVal().getValuesList()),
                Longs.toArray(value.getDoubleVal().getShapeList()));
        }
    }

    @Override
    public BooleanTensor getInitialBoolValue(Object valueKey) {
        KeanuSavedBayesNet.VertexValue value = (KeanuSavedBayesNet.VertexValue)valueKey;

        return extractBoolValue(value);
    }

    @Override
    public IntegerTensor getInitialIntValue(Object valueKey) {
        KeanuSavedBayesNet.VertexValue value = (KeanuSavedBayesNet.VertexValue)valueKey;

        return extractIntValue(value);
    }

    private <T> Vertex<T> fromProtoBuf(KeanuSavedBayesNet.Vertex vertex,
                                       Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        Class vertexClass;
        try {
            vertexClass = Class.forName(vertex.getVertexType());
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Unknown Vertex Type Specified: " + vertex.getVertexType(), e);
        }

        Constructor vertexConstructor;
        try {
            vertexConstructor = vertexClass.getConstructor(Map.class, NetworkReader.class, Object.class);
        } catch (NoSuchMethodException e) {
            throw new
                IllegalArgumentException("Vertex Type doesn't support loading from Proto: " + vertex.getVertexType(), e);
        }

        Vertex newVertex;
        Map<String, Vertex> parentsMap = getParentsMap(vertex, existingVertices);

        try {
            newVertex = (Vertex)vertexConstructor.newInstance(parentsMap, this, vertex.getConstantValue());
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to create Vertex of Type: " + vertex.getVertexType(), e);
        }

        return newVertex;
    }

    private Map<String, Vertex> getParentsMap(KeanuSavedBayesNet.Vertex vertex,
                                                     Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        Map<String, Vertex> parentsMap = new HashMap<>();

        for (KeanuSavedBayesNet.NamedParent namedParent : vertex.getParentsList()) {
            Vertex existingParent = existingVertices.get(namedParent.getId());
            if (existingParent == null) {
                throw new IllegalArgumentException("Invalid Parent Specified: " + namedParent);
            }

            parentsMap.put(namedParent.getName(), existingParent);
        }

        return parentsMap;
    }
}
