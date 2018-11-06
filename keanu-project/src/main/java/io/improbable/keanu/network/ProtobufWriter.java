package io.improbable.keanu.network;

import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.io.IOException;
import java.io.OutputStream;

public class ProtobufWriter {
    private final OutputStream output;
    private final KeanuSavedBayesNet.BayesianNetwork.Builder bayesNetBuilder;

    public ProtobufWriter(OutputStream output) {
        this.output = output;
        bayesNetBuilder = KeanuSavedBayesNet.BayesianNetwork.newBuilder();
    }

    public void save(BayesianNetwork net, boolean saveValues) throws IOException {
        net.save(this);

        if (saveValues) {
            net.saveValues(this);
        }
        bayesNetBuilder.build().writeTo(output);
    }

    public void save(Vertex vertex) {
        KeanuSavedBayesNet.Vertex.Builder vertexBuilder = buildVertex(vertex);
        vertexBuilder.addAllParents(vertex.getNamedParents());
        bayesNetBuilder.addVertices(vertexBuilder.build());
    }

    public void save(ConstantDoubleVertex vertex) {
        KeanuSavedBayesNet.Vertex.Builder vertexBuilder = buildVertex(vertex);
        vertexBuilder.setConstantValue(getValue(vertex).getValue());
        bayesNetBuilder.addVertices(vertexBuilder.build());
    }

    private KeanuSavedBayesNet.Vertex.Builder buildVertex(Vertex vertex) {
        KeanuSavedBayesNet.Vertex.Builder vertexBuilder = KeanuSavedBayesNet.Vertex.newBuilder();

        if (vertex.getLabel() != null) {
            vertexBuilder = vertexBuilder.setLabel(vertex.getLabel().toString());
        }

        vertexBuilder = vertexBuilder.setId(vertex.getId().toProtoBuf());
        vertexBuilder = vertexBuilder.setVertexType(vertex.getClass().getCanonicalName());

        return vertexBuilder;
    }

    public void saveValue(Vertex vertex) {
        //TODO - Remove once we have this everywhere
        throw new UnsupportedOperationException("This Vertex Doesn't Support Value Save");
    }

    public void saveValue(DoubleVertex vertex) {
        KeanuSavedBayesNet.StoredValue storedValue = getValue(vertex);
        bayesNetBuilder.addDefaultState(storedValue);
    }

    private KeanuSavedBayesNet.StoredValue getValue(DoubleVertex vertex) {
        KeanuSavedBayesNet.DoubleTensor savedValue = KeanuSavedBayesNet.DoubleTensor.newBuilder()
            .addAllValues(vertex.getValue().asFlatList())
            .build();

        KeanuSavedBayesNet.VertexValue value = KeanuSavedBayesNet.VertexValue.newBuilder()
            .setDoubleVal(savedValue)
            .build();

        return KeanuSavedBayesNet.StoredValue.newBuilder()
            .setId(vertex.getId().toProtoBuf())
            .setValue(value)
            .build();
    }
}
