package io.improbable.keanu.util.io;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
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
import io.improbable.mir.KeanuSavedBayesNet;
import io.improbable.mir.SavedBayesNet;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public class ProtobufSaver implements NetworkSaver {

    private final BayesianNetwork net;
    private SavedBayesNet.Graph.Builder graphBuilder = null;

    public ProtobufSaver(BayesianNetwork net) {
        this.net = net;
    }
    
    @Override
    public void save(OutputStream output, boolean saveValues, Map<String, String> metadata) throws IOException {
        KeanuSavedBayesNet.ProtoModel protobufModel = getModel(saveValues, metadata);
        protobufModel.writeTo(output);
        graphBuilder = null;
    }

    protected KeanuSavedBayesNet.ProtoModel getModel(boolean withSavedValues, Map<String, String> metadata) {
        SavedBayesNet.Graph graph = getGraph(withSavedValues);
        KeanuSavedBayesNet.ProtoModel.Builder builder = KeanuSavedBayesNet.ProtoModel.newBuilder().setGraph(graph);

        if (metadata != null) {
            builder.setMetadata(buildMetadata(metadata));
        }

        return builder.build();
    }

    protected SavedBayesNet.Graph getGraph(boolean withSavedValues) {
        createGraph(withSavedValues);
        return graphBuilder.build();
    }

    private void createGraph(boolean saveValues) {
        graphBuilder = SavedBayesNet.Graph.newBuilder();

        net.save(this);

        if (saveValues) {
            net.saveValues(this);
        }
    }

    private KeanuSavedBayesNet.ModelMetadata buildMetadata(Map<String, String> metadata) {
        KeanuSavedBayesNet.ModelMetadata.Builder metadataBuilder = KeanuSavedBayesNet.ModelMetadata.newBuilder();
        String[] metadataKeys = metadata.keySet().toArray(new String[0]);
        Arrays.sort(metadataKeys);
        for (String metadataKey : metadataKeys) {
            metadataBuilder.putMetadataInfo(metadataKey, metadata.get(metadataKey));
        }

        return metadataBuilder.build();
    }

    @Override
    public void save(Vertex vertex) {
        if (vertex instanceof NonSaveableVertex) {
            throw new IllegalArgumentException("Trying to save a vertex that isn't Saveable");
        }

        graphBuilder.addVertices(buildVertex(vertex));
    }

    private SavedBayesNet.Vertex buildVertex(Vertex vertex) {
        SavedBayesNet.Vertex.Builder vertexBuilder = SavedBayesNet.Vertex.newBuilder();

        if (vertex.getLabel() != null) {
            vertexBuilder = vertexBuilder.setLabel(vertex.getLabel().toString());
        }

        vertexBuilder = vertexBuilder.setId(SavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()));
        vertexBuilder = vertexBuilder.setVertexType(vertex.getClass().getCanonicalName());
        vertexBuilder = vertexBuilder.addAllShape(Longs.asList(vertex.getShape()));
        saveParams(vertexBuilder, vertex);

        return vertexBuilder.build();
    }

    private void saveParams(SavedBayesNet.Vertex.Builder vertexBuilder,
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

    private SavedBayesNet.NamedParam getEncodedParam(Vertex vertex, String paramName, Method getParamMethod) {
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

    private SavedBayesNet.NamedParam getTypedParam(String paramName, Object param) {
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
        } else if (Boolean.class.isAssignableFrom(param.getClass())) {
            return getParam(paramName, builder -> builder.setBoolParam((boolean) param));
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

    private SavedBayesNet.NamedParam getParam(String paramName,
                                                   Consumer<SavedBayesNet.NamedParam.Builder> valueSetter) {
        SavedBayesNet.NamedParam.Builder paramBuilder = SavedBayesNet.NamedParam.newBuilder();

        paramBuilder.setName(paramName);
        valueSetter.accept(paramBuilder);

        return paramBuilder.build();
    }

    private SavedBayesNet.NamedParam getParam(String paramName, Vertex parent) {
        return getParam(paramName,
            builder -> builder.setParentVertex(
                SavedBayesNet.VertexID.newBuilder().setId(parent.getId().toString())
            )
        );
    }

    private SavedBayesNet.NamedParam getParam(String paramName, long[] param) {
        return getParam(paramName,
            builder -> builder.setLongArrayParam(
                SavedBayesNet.LongArray.newBuilder().addAllValues(Longs.asList(param))));
    }

    private SavedBayesNet.NamedParam getParam(String paramName, int[] param) {
        return getParam(paramName,
            builder -> builder.setIntArrayParam(
                SavedBayesNet.IntArray.newBuilder().addAllValues(Ints.asList(param))));
    }

    private SavedBayesNet.NamedParam getParam(String paramName, Vertex[] param) {
        SavedBayesNet.VertexArray.Builder vertexArray = SavedBayesNet.VertexArray.newBuilder();
        for (Vertex vertex : param) {
            vertexArray.addValues(SavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()));
        }

        return getParam(paramName, builder -> builder.setVertexArrayParam(vertexArray.build()));
    }

    private SavedBayesNet.DoubleTensor getTensor(DoubleTensor tensor) {
        return SavedBayesNet.DoubleTensor.newBuilder()
            .addAllValues(tensor.asFlatList())
            .addAllShape(Longs.asList(tensor.getShape()))
            .build();
    }

    private SavedBayesNet.IntegerTensor getTensor(IntegerTensor tensor) {
        return SavedBayesNet.IntegerTensor.newBuilder()
            .addAllValues(tensor.asFlatList())
            .addAllShape(Longs.asList(tensor.getShape()))
            .build();
    }

    private SavedBayesNet.BooleanTensor getTensor(BooleanTensor tensor) {
        return SavedBayesNet.BooleanTensor.newBuilder()
            .addAllValues(tensor.asFlatList())
            .addAllShape(Longs.asList(tensor.getShape()))
            .build();
    }

    @Override
    public void saveValue(Vertex vertex) {
        if (vertex.hasValue()) {
            SavedBayesNet.StoredValue value = getValue(vertex, vertex.getValue().toString());
            graphBuilder.addDefaultState(value);
        }
    }

    @Override
    public void saveValue(DoubleVertex vertex) {
        if (vertex.hasValue()) {
            SavedBayesNet.StoredValue value = getValue(vertex);
            graphBuilder.addDefaultState(value);
        }
    }

    @Override
    public void saveValue(IntegerVertex vertex) {
        if (vertex.hasValue()) {
            SavedBayesNet.StoredValue value = getValue(vertex);
            graphBuilder.addDefaultState(value);
        }
    }

    @Override
    public void saveValue(BooleanVertex vertex) {
        if (vertex.hasValue()) {
            SavedBayesNet.StoredValue value = getValue(vertex);
            graphBuilder.addDefaultState(value);
        }
    }

    private SavedBayesNet.StoredValue getValue(Vertex vertex, String formattedValue) {
        SavedBayesNet.GenericTensor savedValue = SavedBayesNet.GenericTensor.newBuilder()
            .addAllShape(Longs.asList(vertex.getShape()))
            .addValues(formattedValue)
            .build();

        SavedBayesNet.VertexValue value = SavedBayesNet.VertexValue.newBuilder()
            .setGenericVal(savedValue)
            .build();

        return getStoredValue(vertex, value);

    }

    private SavedBayesNet.StoredValue getValue(DoubleVertex vertex) {
        SavedBayesNet.DoubleTensor savedValue = getTensor(vertex.getValue());

        SavedBayesNet.VertexValue value = SavedBayesNet.VertexValue.newBuilder()
            .setDoubleVal(savedValue)
            .build();

        return getStoredValue(vertex, value);
    }

    private SavedBayesNet.StoredValue getValue(IntegerVertex vertex) {
        SavedBayesNet.IntegerTensor savedValue = getTensor(vertex.getValue());

        SavedBayesNet.VertexValue value = SavedBayesNet.VertexValue.newBuilder()
            .setIntVal(savedValue)
            .build();

        return getStoredValue(vertex, value);
    }

    private SavedBayesNet.StoredValue getValue(BooleanVertex vertex) {
        SavedBayesNet.BooleanTensor savedValue = getTensor(vertex.getValue());

        SavedBayesNet.VertexValue value = SavedBayesNet.VertexValue.newBuilder()
            .setBoolVal(savedValue)
            .build();

        return getStoredValue(vertex, value);
    }

    private SavedBayesNet.StoredValue getStoredValue(Vertex vertex, SavedBayesNet.VertexValue value) {
        return SavedBayesNet.StoredValue.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId(vertex.getId().toString()).build())
            .setValue(value)
            .setIsObserved(vertex.isObserved())
            .build();
    }
}
