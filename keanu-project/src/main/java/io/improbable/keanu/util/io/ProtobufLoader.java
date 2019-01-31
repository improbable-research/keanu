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
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.mir.KeanuSavedBayesNet;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Parameter;
import java.util.HashMap;
import java.util.Map;

public class ProtobufLoader implements NetworkLoader {

    private final Map<Vertex, KeanuSavedBayesNet.StoredValue> savedValues;

    public ProtobufLoader() {
        savedValues = new HashMap<>();
    }

    @Override
    public void loadValue(Vertex vertex) {
        throw new IllegalArgumentException("Cannot Load value for Untyped Vertex");
    }

    @Override
    public BayesianNetwork loadNetwork(InputStream input) throws IOException {
        KeanuSavedBayesNet.Model parsedModel = KeanuSavedBayesNet.Model.parseFrom(input);
        return loadNetwork(parsedModel);
    }

    public BayesianNetwork loadNetwork(KeanuSavedBayesNet.Model parsedModel) {
        Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices = new HashMap<>();

        for (KeanuSavedBayesNet.Vertex vertex : parsedModel.getNetwork().getVerticesList()) {
            Vertex newVertex = createVertexFromProtoBuf(vertex, instantiatedVertices);
            instantiatedVertices.put(vertex.getId(), newVertex);
        }

        BayesianNetwork bayesNet = new BayesianNetwork(instantiatedVertices.values());

        loadDefaultValues(parsedModel.getNetworkState(), instantiatedVertices, bayesNet);

        return bayesNet;
    }

    @Override
    public void loadValue(DoubleVertex vertex) {
        KeanuSavedBayesNet.StoredValue valueInformation = savedValues.get(vertex);
        KeanuSavedBayesNet.VertexValue value = valueInformation.getValue();
        DoubleTensor tensor = extractDoubleValue(value);
        setOrObserveValue(vertex, tensor, valueInformation.getIsObserved());
    }

    private DoubleTensor extractDoubleValue(KeanuSavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.DOUBLEVAL) {
            throw new IllegalArgumentException("Non Double Value specified for Double Vertex");
        } else {
            return extractDoubleTensor(value.getDoubleVal());
        }
    }

    private void loadDefaultValues(KeanuSavedBayesNet.BayesianNetworkState parsedNetworkState,
                                   Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                   BayesianNetwork bayesNet) {
        for (KeanuSavedBayesNet.StoredValue value : parsedNetworkState.getDefaultStateList()) {
            Vertex targetVertex = getTargetVertex(value, instantiatedVertices, bayesNet);

            savedValues.put(targetVertex, value);
            targetVertex.loadValue(this);
        }
    }

    private Vertex getTargetVertex(KeanuSavedBayesNet.StoredValue storedValue,
                                   Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                   BayesianNetwork bayesNet) {
        Vertex idTarget = getTargetByID(storedValue, instantiatedVertices);
        Vertex labelTarget = getTargetByLabel(storedValue, instantiatedVertices, bayesNet);

        return checkTargetsAreValid(idTarget, labelTarget, storedValue);
    }

    private Vertex getTargetByID(KeanuSavedBayesNet.StoredValue storedValue,
                                 Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices) {
        if (storedValue.hasId()) {
            return instantiatedVertices.get(storedValue.getId());
        } else {
            return null;
        }
    }

    private Vertex getTargetByLabel(KeanuSavedBayesNet.StoredValue storedValue,
                                    Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                    BayesianNetwork bayesNet) {
        if (!storedValue.getVertexLabel().isEmpty()) {
            return bayesNet.getVertexByLabel(new VertexLabel(storedValue.getVertexLabel()));
        } else {
            return null;
        }
    }

    private Vertex checkTargetsAreValid(Vertex idTarget, Vertex labelTarget, KeanuSavedBayesNet.StoredValue storedValue) {
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
        KeanuSavedBayesNet.StoredValue valueInformation = savedValues.get(vertex);
        KeanuSavedBayesNet.VertexValue value = valueInformation.getValue();
        BooleanTensor tensor = extractBoolValue(value);
        setOrObserveValue(vertex, tensor, valueInformation.getIsObserved());
    }

    private BooleanTensor extractBoolValue(KeanuSavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.BOOLVAL) {
            throw new IllegalArgumentException("Non Boolean Value specified for Boolean Vertex");
        } else {
            return extractBoolTensor(value.getBoolVal());
        }
    }

    @Override
    public void loadValue(IntegerVertex vertex) {
        KeanuSavedBayesNet.StoredValue valueInformation = savedValues.get(vertex);
        KeanuSavedBayesNet.VertexValue value = valueInformation.getValue();
        IntegerTensor tensor = extractIntValue(value);
        setOrObserveValue(vertex, tensor, valueInformation.getIsObserved());
    }

    private IntegerTensor extractIntValue(KeanuSavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.INTVAL) {
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

    private <T> Vertex<T> createVertexFromProtoBuf(KeanuSavedBayesNet.Vertex vertex,
                                                   Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        Class vertexClass;
        try {
            vertexClass = Class.forName(vertex.getVertexType());
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Unknown Vertex Type Specified: " + vertex.getVertexType(), e);
        }

        Map<String, Object> parameterMap = getParameterMap(vertex, existingVertices);
        Vertex newVertex = instantiateVertex(vertexClass, parameterMap, vertex);

        if (!vertex.getLabel().isEmpty()) {
            newVertex.setLabel(vertex.getLabel());
        }

        return newVertex;
    }

    private Vertex instantiateVertex(Class vertexClass,
                                     Map<String, Object> paramMap,
                                     KeanuSavedBayesNet.Vertex vertex) {
        Constructor<Vertex> loadConstructor = getAnnotatedConstructor(vertexClass);
        Parameter[] constructorParameters = loadConstructor.getParameters();
        Object[] arguments = new Object[constructorParameters.length];

        for (int i = 0; i < constructorParameters.length; i++) {
            arguments[i] = getParameter(constructorParameters[i], paramMap, vertex);

            Class argumentClass = arguments[i].getClass();
            Class parameterClass = Primitives.wrap(constructorParameters[i].getType());

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
                                KeanuSavedBayesNet.Vertex vertex) {
        LoadVertexParam paramAnnotation;

        if ((paramAnnotation = methodParameter.getAnnotation(LoadVertexParam.class)) != null) {
            Object parameter = paramMap.get(paramAnnotation.value());
            if (parameter == null) {
                throw new IllegalArgumentException("Failed to create vertex due to missing parent: "
                    + paramAnnotation.value());
            } else {
                return parameter;
            }
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

    private Map<String, Object> getParameterMap(KeanuSavedBayesNet.Vertex vertex,
                                                Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        Map<String, Object> parameterMap = new HashMap<>();

        for (KeanuSavedBayesNet.NamedParam parameter : vertex.getParametersList()) {
            parameterMap.put(parameter.getName(), getDecodedParam(parameter, existingVertices));
        }

        return parameterMap;
    }

    private Object getDecodedParam(KeanuSavedBayesNet.NamedParam parameter,
                                   Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        switch (parameter.getParamCase()) {
            case PARENTVERTEX:
                return existingVertices.get(parameter.getParentVertex());

            case DOUBLETENSORPARAM:
                return extractDoubleTensor(parameter.getDoubleTensorParam());

            case INTTENSORPARAM:
                return extractIntTensor(parameter.getIntTensorParam());

            case BOOLTENSORPARAM:
                return extractBoolTensor(parameter.getBoolTensorParam());

            case DOUBLEPARAM:
                return parameter.getDoubleParam();

            case INTPARAM:
                return parameter.getIntParam();

            case LONGPARAM:
                return parameter.getLongParam();

            case STRINGPARAM:
                return parameter.getStringParam();

            case BOOLPARAM:
                return parameter.getBoolParam();

            case LONGARRAYPARAM:
                return Longs.toArray(parameter.getLongArrayParam().getValuesList());

            case INTARRAYPARAM:
                return Ints.toArray(parameter.getIntArrayParam().getValuesList());

            case VERTEXARRAYPARAM:
                return extractVertexArray(parameter, existingVertices);

            default:
                throw new IllegalArgumentException("Unknown Param Type Received: "
                    + parameter.getParamCase().toString());
        }
    }

    private Vertex[] extractVertexArray(KeanuSavedBayesNet.NamedParam param,
                                        Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        Vertex[] newVertexArray = new Vertex[param.getVertexArrayParam().getValuesCount()];

        for (int i = 0; i < newVertexArray.length; i++) {
            KeanuSavedBayesNet.VertexID parentId = param.getVertexArrayParam().getValues(i);
            Vertex parentVertex = existingVertices.get(parentId);

            if (parentVertex == null) {
                throw new IllegalArgumentException("Saved Structure references unknown Parent: "
                    + parentId.toString());
            }

            newVertexArray[i] = parentVertex;
        }

        return newVertexArray;
    }

    private DoubleTensor extractDoubleTensor(KeanuSavedBayesNet.DoubleTensor tensor) {
        return DoubleTensor.create(
            Doubles.toArray(tensor.getValuesList()),
            Longs.toArray(tensor.getShapeList()));
    }

    private IntegerTensor extractIntTensor(KeanuSavedBayesNet.IntegerTensor tensor) {
        return IntegerTensor.create(
            Ints.toArray(tensor.getValuesList()),
            Longs.toArray(tensor.getShapeList()));
    }

    private BooleanTensor extractBoolTensor(KeanuSavedBayesNet.BooleanTensor tensor) {
        return BooleanTensor.create(
            Booleans.toArray(tensor.getValuesList()),
            Longs.toArray(tensor.getShapeList()));
    }
}
