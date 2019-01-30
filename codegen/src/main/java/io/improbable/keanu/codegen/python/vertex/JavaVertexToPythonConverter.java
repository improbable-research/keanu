package io.improbable.keanu.codegen.python.vertex;

import com.google.common.collect.ImmutableList;
import com.google.gson.internal.Primitives;
import io.improbable.keanu.codegen.python.DocString;
import io.improbable.keanu.codegen.python.PythonParam;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import org.apache.commons.lang3.NotImplementedException;
import org.reflections.Reflections;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Apache FreeMarker Format:
 * def ${constructor.pythonClass}(${constructor.pythonTypedParameters}) -> Vertex:
 *      ${constructor.docString}return ${constructor.pythonVertexClass}(context.jvm_view().${constructor.javaClass}, ${constructor.pythonParameters})
 */
class JavaVertexToPythonConverter {

    /**
     * Extra parameters are added:
     * - as optional parameters to TypedParams, at the end
     * - as required parameters to CastedParams, before *args
     */
    private static final List<PythonParam> EXTRA_PARAMS = ImmutableList.of(
        new PythonParam("label", String.class, "None")
    );
    private Constructor javaConstructor;
    private Map<String, DocString> docStringMap;
    private List<PythonParam> allParams;

    JavaVertexToPythonConverter(Constructor javaConstructor, Reflections reflections, Map<String, DocString> docStringMap) {
       this.javaConstructor = javaConstructor;
       this.docStringMap = docStringMap;
       this.allParams = createListOfPythonVertexParam(javaConstructor, reflections);
    }

    private List<PythonParam> createListOfPythonVertexParam(Constructor javaConstructor, Reflections reflections) {
        List<String> paramNames = reflections.getConstructorParamNames(javaConstructor);
        List<Class> paramTypes = Arrays.asList(javaConstructor.getParameterTypes());

        List<PythonParam> listOfPythonVertexParams = new ArrayList<>();

        for (int i = 0; i < javaConstructor.getParameterCount(); i++) {
            listOfPythonVertexParams.add(new PythonParam(paramNames.get(i), paramTypes.get(i)));
        }

        return listOfPythonVertexParams;
    }

    String getJavaClass() {
        return javaConstructor.getDeclaringClass().getSimpleName();
    }

    String getPythonClass() {
        return javaConstructor.getDeclaringClass().getSimpleName().replaceAll("Vertex$", "");
    }

    String getPythonVertexClass() {
        Class<?> javaClass = javaConstructor.getDeclaringClass();
        if (DoubleVertex.class.isAssignableFrom(javaClass)) {
            return "Double";
        } else if (IntegerVertex.class.isAssignableFrom(javaClass)) {
            return "Integer";
        } else if (BooleanVertex.class.isAssignableFrom(javaClass)) {
            return "Boolean";
        } else {
            return "Vertex";
        }
    }

    String getPythonTypedParameters() {
        List<String> pythonParams = new ArrayList<>();

        for (PythonParam param : allParams) {
            pythonParams.add(param.getName() + ": " + toTypedParam(param.getKlass()));
        }

        for (PythonParam extraParam : EXTRA_PARAMS) {
            if (!allParams.contains(extraParam)) {
                pythonParams.add(extraParam.getName() + ": Optional[" + toTypedParam(extraParam.getKlass()) + "]=" + extraParam.getDefaultValue());
            }
        }

        return String.join(", ", pythonParams);
    }

    private String toTypedParam(Class<?> parameterClass) {
        Class parameterType = Primitives.wrap(parameterClass);

        if (Vertex.class.isAssignableFrom(parameterType)) {
            return "vertex_constructor_param_types";
        } else if (DoubleTensor.class.isAssignableFrom(parameterType) ||
            IntegerTensor.class.isAssignableFrom(parameterType) ||
            BooleanTensor.class.isAssignableFrom(parameterType)) {
            return "tensor_arg_types";
        } else if (Double.class.isAssignableFrom(parameterType)) {
            return "float";
        } else if (Boolean.class.isAssignableFrom(parameterType)) {
            return "bool";
        } else if (Integer.class.isAssignableFrom(parameterType) || Long.class.isAssignableFrom(parameterType)) {
            return "int";
        } else if (String.class.isAssignableFrom(parameterType)) {
            return "str";
        } else if (Long[].class.isAssignableFrom(parameterType) || Integer[].class.isAssignableFrom(parameterType) ||
            long[].class.isAssignableFrom(parameterType) || int[].class.isAssignableFrom(parameterType)) {
            return "Collection[int]";
        } else if (Vertex[].class.isAssignableFrom(parameterType)) {
            return "Collection[Vertex]";
        } else {
            throw new NotImplementedException(String.format("Mapping from Java type %s is not defined.", parameterType.getName()));
        }
    }

    String getPythonParameters() {
        List<String> pythonParams = new ArrayList<>();

        for (PythonParam extraParam : EXTRA_PARAMS) {
            pythonParams.add(toCastedParam(extraParam.getName(), extraParam.getKlass()));
        }

        for (PythonParam param : allParams) {
            pythonParams.add(toCastedParam(param.getName(), param.getKlass()));
        }

        return String.join(", ", pythonParams);
    }

    private String toCastedParam(String pythonParameter, Class<?> parameterClass) {
        Class parameterType = Primitives.wrap(parameterClass);

        if (DoubleVertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_double_vertex(" + pythonParameter + ")";
        } else if (IntegerVertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_integer_vertex(" + pythonParameter + ")";
        } else if (BooleanVertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_boolean_vertex(" + pythonParameter + ")";
        } else if (Vertex.class.isAssignableFrom(parameterType)) {
            return "cast_to_vertex(" + pythonParameter + ")";
        } else if (DoubleTensor.class.isAssignableFrom(parameterType)) {
            return "cast_to_double_tensor(" + pythonParameter + ")";
        } else if (IntegerTensor.class.isAssignableFrom(parameterType)) {
            return "cast_to_integer_tensor(" + pythonParameter + ")";
        } else if (BooleanTensor.class.isAssignableFrom(parameterType)) {
            return "cast_to_boolean_tensor(" + pythonParameter + ")";
        } else if (Double.class.isAssignableFrom(parameterType)) {
            return "cast_to_double(" + pythonParameter + ")";
        } else if (Integer.class.isAssignableFrom(parameterType) || Long.class.isAssignableFrom(parameterType)) {
            return "cast_to_integer(" + pythonParameter + ")";
        } else if (String.class.isAssignableFrom(parameterType)) {
            return pythonParameter;
        } else if (Boolean.class.isAssignableFrom(parameterType)) {
            return "cast_to_boolean(" + pythonParameter + ")";
        } else if (Long[].class.isAssignableFrom(parameterType) || long[].class.isAssignableFrom(parameterType)) {
            return "cast_to_long_array(" + pythonParameter + ")";
        } else if (Integer[].class.isAssignableFrom(parameterType) || int[].class.isAssignableFrom(parameterType)) {
            return "cast_to_int_array(" + pythonParameter + ")";
        } else if (Vertex[].class.isAssignableFrom(parameterType)) {
            return "cast_to_vertex_array(" + pythonParameter + ")";
        } else {
            throw new IllegalArgumentException("Failed to Encode " + pythonParameter + " of type: " + parameterType);
        }
    }

    String getDocString() {
        return docStringMap.get(javaConstructor.getName()).getAsString();
    }
}