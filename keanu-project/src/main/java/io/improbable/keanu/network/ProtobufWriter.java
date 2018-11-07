package io.improbable.keanu.network;

import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.io.IOException;
import java.io.OutputStream;

public class ProtobufWriter implements NetworkWriter {
    private final BayesianNetwork net;
    private KeanuSavedBayesNet.BayesianNetwork.Builder bayesNetBuilder = null;

    public ProtobufWriter(BayesianNetwork net) {
        this.net = net;
    }

    public void save(OutputStream output, boolean saveValues) throws IOException {
        bayesNetBuilder = KeanuSavedBayesNet.BayesianNetwork.newBuilder();

        net.save(this);

        if (saveValues) {
            net.saveValues(this);
        }

        bayesNetBuilder.build().writeTo(output);
        bayesNetBuilder = null;
    }

    public void save(Vertex vertex) {
        KeanuSavedBayesNet.Vertex.Builder vertexBuilder = buildVertex(vertex);
        vertexBuilder.addAllParents(vertex.getParentsAsProto());
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

        vertexBuilder = vertexBuilder.setId(KeanuSavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()));
        vertexBuilder = vertexBuilder.setVertexType(vertex.getClass().getCanonicalName());

        return vertexBuilder;
    }

    public void saveValue(Vertex vertex) {
        //TODO - Remove once we have a version for all Vertex data types
        throw new UnsupportedOperationException("This Vertex Doesn't Support Value Save");
    }

    public void saveValue(DoubleVertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue storedValue = getValue(vertex);
            bayesNetBuilder.addDefaultState(storedValue);
        }
    }

    private KeanuSavedBayesNet.StoredValue getValue(DoubleVertex vertex) {
        KeanuSavedBayesNet.DoubleTensor savedValue = KeanuSavedBayesNet.DoubleTensor.newBuilder()
            .addAllValues(vertex.getValue().asFlatList())
            .addAllShape(Longs.asList(vertex.getShape()))
            .build();

        KeanuSavedBayesNet.VertexValue value = KeanuSavedBayesNet.VertexValue.newBuilder()
            .setDoubleVal(savedValue)
            .build();

        return KeanuSavedBayesNet.StoredValue.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()))
            .setValue(value)
            .build();
    }
}
