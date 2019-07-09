package io.improbable.keanu.util.io;

import com.google.common.primitives.Booleans;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.gson.internal.Primitives;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.mir.KeanuSavedBayesNet;
import io.improbable.mir.SavedBayesNet;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Parameter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProtobufLoader implements NetworkLoader {

    private final Map<Vertex, SavedBayesNet.StoredValue> savedValues;

    public ProtobufLoader() {
        savedValues = new HashMap<>();
    }

    @Override
    public void loadValue(Vertex vertex) {
        throw new IllegalArgumentException("Cannot Load value for Untyped Vertex");
    }

    @Override
    public BayesianNetwork loadNetwork(InputStream input) throws IOException {
        KeanuSavedBayesNet.ProtoModel parsedModel = KeanuSavedBayesNet.ProtoModel.parseFrom(input);
        return loadNetwork(parsedModel);
    }

    public BayesianNetwork loadNetwork(KeanuSavedBayesNet.ProtoModel parsedModel) {
        return loadNetwork(parsedModel.getGraph());
    }

    protected BayesianNetwork loadNetwork(SavedBayesNet.Graph graph) {
        Map<SavedBayesNet.VertexID, Vertex> instantiatedVertices = new HashMap<>();

        for (SavedBayesNet.Vertex vertex : graph.getVerticesList()) {
            Vertex newVertex = createVertexFromProtoBuf(vertex, instantiatedVertices);
            instantiatedVertices.put(vertex.getId(), newVertex);
        }

        BayesianNetwork bayesNet = new BayesianNetwork(instantiatedVertices.values());

        loadDefaultValues(graph.getDefaultStateList(), instantiatedVertices, bayesNet);

        return bayesNet;
    }

    @Override
    public void loadValue(DoubleVertex vertex) {
        SavedBayesNet.StoredValue valueInformation = savedValues.get(vertex);
        SavedBayesNet.VertexValue value = valueInformation.getValue();
        DoubleTensor tensor = extractDoubleValue(value);
        setOrObserveValue(vertex, tensor, valueInformation.getIsObserved());
    }

    private DoubleTensor extractDoubleValue(SavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != SavedBayesNet.VertexValue.ValueTypeCase.DOUBLE_VAL) {
            throw new IllegalArgumentException("Non Double Value specified for Double Vertex");
        } else {
            return extractDoubleTensor(value.getDoubleVal());
        }
    }

    private void loadDefaultValues(List<SavedBayesNet.StoredValue> defaultState,
                                   Map<SavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                   BayesianNetwork bayesNet) {
        for (SavedBayesNet.StoredValue value : defaultState) {
            Vertex targetVertex = getTargetVertex(value, instantiatedVertices, bayesNet);

            savedValues.put(targetVertex, value);
            targetVertex.loadValue(this);
        }
    }

    private Vertex getTargetVertex(SavedBayesNet.StoredValue storedValue,
                                   Map<SavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                   BayesianNetwork bayesNet) {
        Vertex idTarget = getTargetByID(storedValue, instantiatedVertices);
        Vertex labelTarget = getTargetByLabel(storedValue, instantiatedVertices, bayesNet);

        return checkTargetsAreValid(idTarget, labelTarget, storedValue);
    }

    private Vertex getTargetByID(SavedBayesNet.StoredValue storedValue,
                                 Map<SavedBayesNet.VertexID, Vertex> instantiatedVertices) {
        if (storedValue.hasId()) {
            return instantiatedVertices.get(storedValue.getId());
        } else {
            return null;
        }
    }

    private Vertex getTargetByLabel(SavedBayesNet.StoredValue storedValue,
                                    Map<SavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                    BayesianNetwork bayesNet) {
        if (!storedValue.getVertexLabel().isEmpty()) {
            return bayesNet.getVertexByLabel(new VertexLabel(storedValue.getVertexLabel()));
        } else {
            return null;
        }
    }

    private Vertex checkTargetsAreValid(Vertex idTarget, Vertex labelTarget, SavedBayesNet.StoredValue storedValue) {
        Vertex targetVertex;

        if (idTarget != null && labelTarget != null) {
            if (idTarget != labelTarget) {
                throw new IllegalArgumentException("Label and VertexID don't refer to same Vertex: ("
                    + storedValue.getVertexLabel() + ") ("
                    + storedValue.getId().toString() + ")");
            } else {
                targetVertex = idTarget;
            }
        } else if (idTarget == null && labelTarget == null) {
            throw new IllegalArgumentException("Value specified for unknown Vertex: ("
                + storedValue.getVertexLabel() + ") ("
                + storedValue.getId() + ")");
        } else {
            targetVertex = (idTarget != null) ? idTarget : labelTarget;
        }

        return targetVertex;
    }

    @Override
    public void loadValue(BooleanVertex vertex) {
        SavedBayesNet.StoredValue valueInformation = savedValues.get(vertex);
        SavedBayesNet.VertexValue value = valueInformation.getValue();
        BooleanTensor tensor = extractBoolValue(value);
        setOrObserveValue(vertex, tensor, valueInformation.getIsObserved());
    }

    private BooleanTensor extractBoolValue(SavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != SavedBayesNet.VertexValue.ValueTypeCase.BOOL_VAL) {
            throw new IllegalArgumentException("Non Boolean Value specified for Boolean Vertex");
        } else {
            return extractBoolTensor(value.getBoolVal());
        }
    }

    @Override
    public void loadValue(IntegerVertex vertex) {
        SavedBayesNet.StoredValue valueInformation = savedValues.get(vertex);
        SavedBayesNet.VertexValue value = valueInformation.getValue();
        IntegerTensor tensor = extractIntValue(value);
        setOrObserveValue(vertex, tensor, valueInformation.getIsObserved());
    }

    private IntegerTensor extractIntValue(SavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != SavedBayesNet.VertexValue.ValueTypeCase.INT_VAL) {
            throw new IllegalArgumentException("Non Int Value specified for Int Vertex");
        } else {
            return extractIntTensor(value.getIntVal());
        }
    }

    private void setOrObserveValue(Vertex vertex, Tensor valueTensor, boolean isObserved) {
        if (isObserved) {
            vertex.observe(valueTensor);
        } else {
            vertex.setValue(valueTensor);
        }
    }

    private <T> Vertex<T> createVertexFromProtoBuf(SavedBayesNet.Vertex vertex,
                                                   Map<SavedBayesNet.VertexID, Vertex> existingVertices) {
        Class vertexClass;
        try {
            vertexClass = Class.forName(vertex.getVertexType());
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Unknown Vertex Type Specified: " + vertex.getVertexType(), e);
        }

        Map<String, Object> parameterMap = getParameterMap(vertex, existingVertices);
        Vertex newVertex = instantiateVertex(vertexClass, parameterMap, vertex);

        if (!vertex.getLabel().isEmpty() && !(newVertex instanceof ProxyVertex)) {
            VertexLabel newLabel = VertexLabel.parseLabel(vertex.getLabel());
            newVertex.setLabel(newLabel);
        }

        return newVertex;
    }

    private Vertex instantiateVertex(Class vertexClass,
                                     Map<String, Object> paramMap,
                                     SavedBayesNet.Vertex vertex) {
        Constructor<Vertex> loadConstructor = getAnnotatedConstructor(vertexClass);
        Parameter[] constructorParameters = loadConstructor.getParameters();
        Object[] arguments = new Object[constructorParameters.length];

        for (int i = 0; i < constructorParameters.length; i++) {
            Object parameter = getParameter(constructorParameters[i], paramMap, vertex);
            arguments[i] = parameter;

            Class argumentClass;
            Class parameterClass = Primitives.wrap(constructorParameters[i].getType());
            if (parameter != null) {
                argumentClass = arguments[i].getClass();
            } else {
                argumentClass = parameterClass;
            }

            if (!parameterClass.isAssignableFrom(argumentClass)) {
                throw new IllegalArgumentException("Incorrect Parameter Type specified.  Got: "
                    + argumentClass + ", Expected: " + parameterClass);
            }
        }

        try {
            return loadConstructor.newInstance(arguments);
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to create new Vertex", e);
        }
    }

    private Object getParameter(Parameter methodParameter,
                                Map<String, Object> paramMap,
                                SavedBayesNet.Vertex vertex) {
        LoadVertexParam paramAnnotation;

        if ((paramAnnotation = methodParameter.getAnnotation(LoadVertexParam.class)) != null) {
            Object parameter = paramMap.get(paramAnnotation.value());
            if (parameter == null && !paramAnnotation.isNullable()) {
                throw new IllegalArgumentException("Failed to create vertex due to missing parameter: "
                    + paramAnnotation.value());
            }

            return parameter;
        } else if (methodParameter.getAnnotation(LoadShape.class) != null) {
            if (vertex.getShapeCount() == 0) {
                return Tensor.SCALAR_SHAPE;
            } else {
                return Longs.toArray(vertex.getShapeList());
            }
        } else {
            throw new IllegalArgumentException("Cannot create Vertex due to unannotated parameter in constructor");
        }
    }

    private Constructor getAnnotatedConstructor(Class vertexClass) {
        Constructor[] constructors = vertexClass.getConstructors();

        for (Constructor constructor : constructors) {
            Parameter[] parameters = constructor.getParameters();

            if (parameters.length > 0 &&
                (parameters[0].isAnnotationPresent(LoadVertexParam.class)
                    || parameters[0].isAnnotationPresent(LoadShape.class))) {
                return constructor;
            }
        }

        throw new IllegalArgumentException("No Annotated Load Constructor for Vertex of type: " + vertexClass);
    }

    private Map<String, Object> getParameterMap(SavedBayesNet.Vertex vertex,
                                                Map<SavedBayesNet.VertexID, Vertex> existingVertices) {
        Map<String, Object> parameterMap = new HashMap<>();

        for (SavedBayesNet.NamedParam parameter : vertex.getParametersList()) {
            parameterMap.put(parameter.getName(), getDecodedParam(parameter, existingVertices));
        }

        return parameterMap;
    }

    private Object getDecodedParam(SavedBayesNet.NamedParam parameter,
                                   Map<SavedBayesNet.VertexID, Vertex> existingVertices) {
        switch (parameter.getParamCase()) {
            case PARENT_VERTEX:
                return existingVertices.get(parameter.getParentVertex());

            case DOUBLE_TENSOR_PARAM:
                return extractDoubleTensor(parameter.getDoubleTensorParam());

            case INT_TENSOR_PARAM:
                return extractIntTensor(parameter.getIntTensorParam());

            case BOOL_TENSOR_PARAM:
                return extractBoolTensor(parameter.getBoolTensorParam());

            case DOUBLE_PARAM:
                return parameter.getDoubleParam();

            case INT_PARAM:
                return parameter.getIntParam();

            case LONG_PARAM:
                return parameter.getLongParam();

            case STRING_PARAM:
                return parameter.getStringParam();

            case BOOL_PARAM:
                return parameter.getBoolParam();

            case LONG_ARRAY_PARAM:
                return Longs.toArray(parameter.getLongArrayParam().getValuesList());

            case INT_ARRAY_PARAM:
                return Ints.toArray(parameter.getIntArrayParam().getValuesList());

            case VERTEX_ARRAY_PARAM:
                return extractVertexArray(parameter, existingVertices);

            default:
                throw new IllegalArgumentException("Unknown Param Type Received: "
                    + parameter.getParamCase().toString());
        }
    }

    private Vertex[] extractVertexArray(SavedBayesNet.NamedParam param,
                                        Map<SavedBayesNet.VertexID, Vertex> existingVertices) {
        Vertex[] newVertexArray = new Vertex[param.getVertexArrayParam().getValuesCount()];

        for (int i = 0; i < newVertexArray.length; i++) {
            SavedBayesNet.VertexID parentId = param.getVertexArrayParam().getValues(i);
            Vertex parentVertex = existingVertices.get(parentId);

            if (parentVertex == null) {
                throw new IllegalArgumentException("Saved Structure references unknown Parent: "
                    + parentId.toString());
            }

            newVertexArray[i] = parentVertex;
        }

        return newVertexArray;
    }

    private DoubleTensor extractDoubleTensor(SavedBayesNet.DoubleTensor tensor) {
        return DoubleTensor.create(
            Doubles.toArray(tensor.getValuesList()),
            Longs.toArray(tensor.getShapeList()));
    }

    private IntegerTensor extractIntTensor(SavedBayesNet.IntegerTensor tensor) {
        return IntegerTensor.create(
            Ints.toArray(tensor.getValuesList()),
            Longs.toArray(tensor.getShapeList()));
    }

    private BooleanTensor extractBoolTensor(SavedBayesNet.BooleanTensor tensor) {
        return BooleanTensor.create(
            Booleans.toArray(tensor.getValuesList()),
            Longs.toArray(tensor.getShapeList()));
    }
}
