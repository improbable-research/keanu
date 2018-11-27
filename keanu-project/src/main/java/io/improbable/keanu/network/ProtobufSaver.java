package io.improbable.keanu.network;

import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Method;

public class ProtobufSaver implements NetworkSaver {
    private final BayesianNetwork net;
    private KeanuSavedBayesNet.BayesianNetwork.Builder bayesNetBuilder = null;

    public ProtobufSaver(BayesianNetwork net) {
        this.net = net;
    }

    @Override
    public void save(OutputStream output, boolean saveValues) throws IOException {
        bayesNetBuilder = KeanuSavedBayesNet.BayesianNetwork.newBuilder();

        net.save(this);

        if (saveValues) {
            net.saveValues(this);
        }

        bayesNetBuilder.build().writeTo(output);
        bayesNetBuilder = null;
    }

    @Override
    public void save(Vertex vertex) {
        if (vertex instanceof NonSaveableVertex) {
            throw new IllegalArgumentException("Trying to save a vertex that isn't Saveable");
        }

        KeanuSavedBayesNet.Vertex.Builder vertexBuilder = buildVertex(vertex);
        bayesNetBuilder.addVertices(vertexBuilder.build());
    }

    private KeanuSavedBayesNet.Vertex.Builder buildVertex(Vertex vertex) {
        KeanuSavedBayesNet.Vertex.Builder vertexBuilder = KeanuSavedBayesNet.Vertex.newBuilder();

        if (vertex.getLabel() != null) {
            vertexBuilder = vertexBuilder.setLabel(vertex.getLabel().toString());
        }

        vertexBuilder = vertexBuilder.setId(KeanuSavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()));
        vertexBuilder = vertexBuilder.setVertexType(vertex.getClass().getCanonicalName());
        saveParams(vertexBuilder, vertex);

        return vertexBuilder;
    }

    private void saveParams(KeanuSavedBayesNet.Vertex.Builder vertexBuilder,
                            Vertex vertex) {
        Class vertexClass = vertex.getClass();
        Method[] methods = vertexClass.getMethods();

        for (Method method : methods) {
            SaveVertexParam vertexAnnotation = method.getAnnotation(SaveVertexParam.class);
            if (vertexAnnotation != null) {
                String parentName = vertexAnnotation.value();
                vertexBuilder.addParameters(getEncodedParam(vertex, parentName, method));
            }
        }
    }

    private KeanuSavedBayesNet.NamedParam getEncodedParam(Vertex vertex, String paramName, Method getParamMethod) {
        Object param;

        try {
            param = getParamMethod.invoke(vertex);
        } catch (Exception e) {
            throw new IllegalArgumentException("Invalid parent retrieval function specified", e);
        }

        if (param == null) {
            throw new IllegalArgumentException("No value returned from Save Function");
        }

        return getTypedParam(paramName, param);
    }

    private KeanuSavedBayesNet.NamedParam getTypedParam(String paramName, Object param) {
        if (Vertex.class.isAssignableFrom(param.getClass()))  {
            return getParam(paramName, (Vertex)param);
        } else if (DoubleTensor.class.isAssignableFrom(param.getClass())){
            return getParam(paramName, (DoubleTensor) param);
        } else if (IntegerTensor.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (IntegerTensor) param);
        } else if (BooleanTensor.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (BooleanTensor) param);
        } else if (double.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (double) param);
        } else if (int.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (int) param);
        } else if (long.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (long) param);
        } else if (String.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (String) param);
        } else {
            throw new IllegalArgumentException("Unknown Parameter Type to Save: " + param.getClass().toString());
        }
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, Vertex parent) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setParentVertex(KeanuSavedBayesNet.VertexID.newBuilder().setId(parent.getId().toString()));

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, DoubleTensor param) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setDoubleTensorParam(getTensor(param));

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, IntegerTensor param) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setIntTensorParam(getTensor(param));

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, BooleanTensor param) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setBoolTensorParam(getTensor(param));

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, double param) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setDoubleParam(param);

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, String param) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setStringParam(param);

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, int param) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setIntParam(param);

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, long param) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        paramBuilder.setLongParam(param);

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.DoubleTensor getTensor(DoubleTensor tensor) {
        return KeanuSavedBayesNet.DoubleTensor.newBuilder()
            .addAllValues(tensor.asFlatList())
            .addAllShape(Longs.asList(tensor.getShape()))
            .build();
    }

    private KeanuSavedBayesNet.IntegerTensor getTensor(IntegerTensor tensor) {
        return KeanuSavedBayesNet.IntegerTensor.newBuilder()
            .addAllValues(tensor.asFlatList())
            .addAllShape(Longs.asList(tensor.getShape()))
            .build();
    }

    private KeanuSavedBayesNet.BooleanTensor getTensor(BooleanTensor tensor) {
        return KeanuSavedBayesNet.BooleanTensor.newBuilder()
            .addAllValues(tensor.asFlatList())
            .addAllShape(Longs.asList(tensor.getShape()))
            .build();
    }

    @Override
    public void saveValue(Vertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue value = getValue(vertex, vertex.getValue().toString());
            bayesNetBuilder.addDefaultState(value);
        }
    }

    @Override
    public void saveValue(DoubleVertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue value = getValue(vertex);
            bayesNetBuilder.addDefaultState(value);
        }
    }

    @Override
    public void saveValue(IntegerVertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue value = getValue(vertex);
            bayesNetBuilder.addDefaultState(value);
        }
    }

    @Override
    public void saveValue(BoolVertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue value = getValue(vertex);
            bayesNetBuilder.addDefaultState(value);
        }
    }

    private KeanuSavedBayesNet.StoredValue getValue(Vertex vertex, String formattedValue) {
        KeanuSavedBayesNet.GenericTensor savedValue = KeanuSavedBayesNet.GenericTensor.newBuilder()
            .addAllShape(Longs.asList(vertex.getShape()))
            .addValues(formattedValue)
            .build();

        KeanuSavedBayesNet.VertexValue value = KeanuSavedBayesNet.VertexValue.newBuilder()
            .setGenericVal(savedValue)
            .build();

        return getStoredValue(vertex, value);

    }

    private KeanuSavedBayesNet.StoredValue getValue(DoubleVertex vertex) {
        KeanuSavedBayesNet.DoubleTensor savedValue = getTensor(vertex.getValue());

        KeanuSavedBayesNet.VertexValue value = KeanuSavedBayesNet.VertexValue.newBuilder()
            .setDoubleVal(savedValue)
            .build();

        return getStoredValue(vertex, value);
    }

    private KeanuSavedBayesNet.StoredValue getValue(IntegerVertex vertex) {
        KeanuSavedBayesNet.IntegerTensor savedValue = getTensor(vertex.getValue());

        KeanuSavedBayesNet.VertexValue value = KeanuSavedBayesNet.VertexValue.newBuilder()
            .setIntVal(savedValue)
            .build();

        return getStoredValue(vertex, value);
    }

    private KeanuSavedBayesNet.StoredValue getValue(BoolVertex vertex) {
        KeanuSavedBayesNet.BooleanTensor savedValue = getTensor(vertex.getValue());

        KeanuSavedBayesNet.VertexValue value = KeanuSavedBayesNet.VertexValue.newBuilder()
            .setBoolVal(savedValue)
            .build();

        return getStoredValue(vertex, value);
    }

    private KeanuSavedBayesNet.StoredValue getStoredValue(Vertex vertex, KeanuSavedBayesNet.VertexValue value) {
        return KeanuSavedBayesNet.StoredValue.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()).build())
            .setValue(value)
            .build();
    }
}
