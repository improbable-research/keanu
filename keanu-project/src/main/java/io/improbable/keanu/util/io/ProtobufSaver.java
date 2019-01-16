package io.improbable.keanu.util.io;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public class ProtobufSaver implements NetworkSaver {

    private final BayesianNetwork net;
    private KeanuSavedBayesNet.Model.Builder modelBuilder = null;

    public ProtobufSaver(BayesianNetwork net) {
        this.net = net;
    }

    protected KeanuSavedBayesNet.Model getModel(boolean withSavedValues, Map<String, String> metadata) {
        createProtobufModel(withSavedValues, metadata);
        return modelBuilder.build();
    }

    @Override
    public void save(OutputStream output, boolean saveValues, Map<String, String> metadata) throws IOException {
        KeanuSavedBayesNet.Model protobufModel = getModel(saveValues, metadata);
        protobufModel.writeTo(output);
        modelBuilder = null;
    }

    private void createProtobufModel(boolean saveValues, Map<String, String> metadata) {
        modelBuilder = KeanuSavedBayesNet.Model.newBuilder();

        net.save(this);

        if (saveValues) {
            net.saveValues(this);
        }
        saveMetadata(metadata);
    }

    private void saveMetadata(Map<String, String> metadata) {
        if (metadata != null) {
            KeanuSavedBayesNet.Metadata.Builder metadataBuilder = KeanuSavedBayesNet.Metadata.newBuilder();
            String[] metadataKeys = metadata.keySet().toArray(new String[0]);
            Arrays.sort(metadataKeys);
            for (String metadataKey : metadataKeys) {
                metadataBuilder.putMetadataInfo(metadataKey, metadata.get(metadataKey));
            }
            modelBuilder.setMetadata(metadataBuilder);
        }
    }

    @Override
    public void save(Vertex vertex) {
        if (vertex instanceof NonSaveableVertex) {
            throw new IllegalArgumentException("Trying to save a vertex that isn't Saveable");
        }

        modelBuilder.getNetworkBuilder().addVertices(buildVertex(vertex));
    }

    private KeanuSavedBayesNet.Vertex buildVertex(Vertex vertex) {
        KeanuSavedBayesNet.Vertex.Builder vertexBuilder = KeanuSavedBayesNet.Vertex.newBuilder();

        if (vertex.getLabel() != null) {
            vertexBuilder = vertexBuilder.setLabel(vertex.getLabel().toString());
        }

        vertexBuilder = vertexBuilder.setId(KeanuSavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()));
        vertexBuilder = vertexBuilder.setVertexType(vertex.getClass().getCanonicalName());
        vertexBuilder = vertexBuilder.addAllShape(Longs.asList(vertex.getShape()));
        saveParams(vertexBuilder, vertex);

        return vertexBuilder.build();
    }

    private void saveParams(KeanuSavedBayesNet.Vertex.Builder vertexBuilder,
                            Vertex vertex) {
        Map<String, Method> parentRetrievalMethodMap = getParentRetrievalMethodMap(vertex);

        String[] parentNames = parentRetrievalMethodMap.keySet().toArray(new String[0]);
        Arrays.sort(parentNames);
        for (String parentName : parentNames) {
            vertexBuilder.addParameters(getEncodedParam(vertex, parentName, parentRetrievalMethodMap.get(parentName)));
        }
    }

    private Map<String, Method> getParentRetrievalMethodMap(Vertex vertex) {
        Class vertexClass = vertex.getClass();
        Method[] methods = vertexClass.getMethods();
        Map<String, Method> parentRetrievalMethodMap = new HashMap<>();

        for (Method method : methods) {
            SaveVertexParam vertexAnnotation = method.getAnnotation(SaveVertexParam.class);
            if (vertexAnnotation != null) {
                String parentName = vertexAnnotation.value();
                parentRetrievalMethodMap.put(parentName, method);
            }
        }

        return parentRetrievalMethodMap;
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
            return getParam(paramName, builder -> builder.setDoubleTensorParam(getTensor((DoubleTensor) param)));
        } else if (IntegerTensor.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, builder -> builder.setIntTensorParam(getTensor((IntegerTensor) param)));
        } else if (BooleanTensor.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, builder -> builder.setBoolTensorParam(getTensor((BooleanTensor) param)));
        } else if (Double.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, builder -> builder.setDoubleParam((double) param));
        } else if (Integer.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, builder -> builder.setIntParam((int) param));
        } else if (Long.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, builder -> builder.setLongParam((long) param));
        } else if (String.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, builder -> builder.setStringParam((String) param));
        } else if (Long[].class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (long[]) param);
        } else if (Vertex[].class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (Vertex[]) param);
        } else if (Integer[].class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, (int[]) param);
        } else {
            throw new IllegalArgumentException("Unknown Parameter Type to Save: " + param.getClass().toString());
        }
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName,
                                                   Consumer<KeanuSavedBayesNet.NamedParam.Builder> valueSetter) {
        KeanuSavedBayesNet.NamedParam.Builder paramBuilder = KeanuSavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        valueSetter.accept(paramBuilder);

        return paramBuilder.build();
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, Vertex parent) {
        return getParam(paramName,
            builder -> builder.setParentVertex(
                KeanuSavedBayesNet.VertexID.newBuilder().setId(parent.getId().toString())
            )
        );
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, long[] param) {
        return getParam(paramName,
            builder -> builder.setLongArrayParam(
                KeanuSavedBayesNet.LongArray.newBuilder().addAllValues(Longs.asList(param))));
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, int[] param) {
        return getParam(paramName,
            builder -> builder.setIntArrayParam(
                KeanuSavedBayesNet.IntArray.newBuilder().addAllValues(Ints.asList(param))));
    }

    private KeanuSavedBayesNet.NamedParam getParam(String paramName, Vertex[] param) {
        KeanuSavedBayesNet.VertexArray.Builder vertexArray = KeanuSavedBayesNet.VertexArray.newBuilder();
        for (Vertex vertex : param) {
            vertexArray.addValues(KeanuSavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()));
        }

        return getParam(paramName, builder -> builder.setVertexArrayParam(vertexArray.build()));
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
            modelBuilder.getNetworkStateBuilder().addDefaultState(value);
        }
    }

    @Override
    public void saveValue(DoubleVertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue value = getValue(vertex);
            modelBuilder.getNetworkStateBuilder().addDefaultState(value);
        }
    }

    @Override
    public void saveValue(IntegerVertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue value = getValue(vertex);
            modelBuilder.getNetworkStateBuilder().addDefaultState(value);
        }
    }

    @Override
    public void saveValue(BooleanVertex vertex) {
        if (vertex.hasValue()) {
            KeanuSavedBayesNet.StoredValue value = getValue(vertex);
            modelBuilder.getNetworkStateBuilder().addDefaultState(value);
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

    private KeanuSavedBayesNet.StoredValue getValue(BooleanVertex vertex) {
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
            .setIsObserved(vertex.isObserved())
            .build();
    }
}
