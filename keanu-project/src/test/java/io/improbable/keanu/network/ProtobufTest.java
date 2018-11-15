package io.improbable.keanu.network;

import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.LoadVertexValue;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.hamcrest.collection.IsEmptyCollection;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.reflections.Reflections;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.AnnotatedElement;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.hasKey;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;

public class ProtobufTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void youCanSaveAndLoadANetworkWithValues() throws IOException {
        DoubleVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        BayesianNetwork net = new BayesianNetwork(gaussianVertex.getConnectedGraph());
        ByteArrayOutputStream output = new ByteArrayOutputStream();

        ProtobufWriter protobufWriter = new ProtobufWriter(net);
        protobufWriter.save(output, true);
        assertThat(output.size(), greaterThan(0));
        ByteArrayInputStream input = new ByteArrayInputStream(output.toByteArray());

        ProtobufReader reader = new ProtobufReader();
        BayesianNetwork readNet = reader.loadNetwork(input);

        assertThat(readNet.getLatentVertices().size(), is(1));
        assertThat(readNet.getLatentVertices().get(0), instanceOf(GaussianVertex.class));
        GaussianVertex readGaussianVertex = (GaussianVertex)readNet.getLatentVertices().get(0);
        assertThat(readGaussianVertex.getMu().getValue().scalar(), closeTo(0.0, 1e-10));
        assertThat(readGaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        readGaussianVertex.sample();

    }

    @Test
    public void loadFailsIfParentsAreMissing() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex due to missing parent: sigma");

        KeanuSavedBayesNet.Vertex muVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setLabel("MU VERTEX")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .setConstantValue(KeanuSavedBayesNet.VertexValue.newBuilder()
                .setDoubleVal(KeanuSavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1)).addValues(1.0).build()
                ).build())
            .build();

        KeanuSavedBayesNet.Vertex gaussianVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("2"))
            .setLabel("GAUSSIAN VERTEX")
            .setVertexType(GaussianVertex.class.getCanonicalName())
            .addParents(KeanuSavedBayesNet.NamedParent.newBuilder()
                .setName("mu")
                .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1").build())
                .build()
            )
            .build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(muVertex)
            .addVertices(gaussianVertex).build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);

        ProtobufReader reader = new ProtobufReader();
        BayesianNetwork net = reader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfNoConstantSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex as required initial value not present: ");

        KeanuSavedBayesNet.Vertex constantVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setLabel("MU VERTEX")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(constantVertex).build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);
        ProtobufReader reader = new ProtobufReader();
        BayesianNetwork readNet = reader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void allSaveableVerticesHaveCorrectAnnotations() {
        Reflections reflections = new Reflections("io.improbable.keanu.vertices");

        Set<Class<? extends Vertex>> vertices = reflections.getSubTypesOf(Vertex.class);
        vertices.stream()
            .filter(v -> SaveableVertex.class.isAssignableFrom(v))
            .forEach(this::checkSaveableVertex);
    }

    private void checkSaveableVertex(Class<? extends Vertex> vertexClass) {
        if (ConstantVertex.class.isAssignableFrom(vertexClass)) {
            checkRootVertex(vertexClass);
        } else {
            checkNonRootVertex(vertexClass);
        }
    }

    private <A extends AnnotatedElement> List<A> filterAnnotatedObjects(A[] items, Class annotation) {
        List<A> filteredList = new ArrayList<>();

        Arrays.stream(items)
            .filter(item -> item.isAnnotationPresent(annotation))
            .forEach(item -> filteredList.add((A)item));

        return filteredList;
    }

    private List<Constructor> getConstructorsWithAnnotatedParameters(Class parentClass, Class annotation) {
        return Arrays.stream(parentClass.getConstructors())
            .filter(constructor -> !filterAnnotatedObjects(constructor.getParameters(), annotation).isEmpty())
            .collect(Collectors.toList());
    }

    private void checkRootVertex(Class<? extends Vertex> vertexClass) {
        /*
         * For a root Vertex Class we need to check that there is a constructor that takes a value, and no other
         * annotations.
         */
        assertThat("Root classes must not have any Parent Constructors: " + vertexClass,
            getConstructorsWithAnnotatedParameters(vertexClass, LoadParentVertex.class), IsEmptyCollection.empty());

        List<Constructor> loadValueConstructors = getConstructorsWithAnnotatedParameters(vertexClass,
            LoadVertexValue.class);
        assertThat("Root class must have a single value constructor: " + vertexClass,
            loadValueConstructors.size(), is(1));

        assertThat("Root class must have a singleton Constructor: " + vertexClass,
            loadValueConstructors.get(0).getParameterCount(), is(1));
    }

    private void checkNonRootVertex(Class<? extends Vertex> vertexClass) {
        /*
         * For a non-root vertex we need to check that we have a constructor that is annotated with the same things as
         * our parents (and that all parameters are labelled with a valid input).
         */
        assertThat("Non-root Class must not have value constructors: " + vertexClass,
            getConstructorsWithAnnotatedParameters(vertexClass, LoadVertexValue.class), IsEmptyCollection.empty());

        Map<String, Class> storedParams = getSavedParams(vertexClass);
        Map<String, Class> requiredParams = getRequiredParamsAndCheckOnlyUsedOnce(vertexClass);

        for (Map.Entry<String, Class> param : requiredParams.entrySet()) {
            assertThat("Class must save all required params: " + vertexClass,
                storedParams, hasKey(param.getKey()));
            assertThat("Saved and Loaded Param " + param.getKey() + " must have same type",
                param.getValue() == storedParams.get(param.getKey()));
        }
    }

    private Map<String, Class> getSavedParams(Class<? extends Vertex> vertexClass) {
        Map<String, Class> savedParams = new HashMap<>();

        for (Method method : filterAnnotatedObjects(vertexClass.getMethods(), SaveParentVertex.class)) {
            String paramName = method.getAnnotation(SaveParentVertex.class).value();
            Class paramType = method.getReturnType();
            savedParams.put(paramName, paramType);
        }

        return savedParams;
    }

    private Map<String, Class> getRequiredParamsAndCheckOnlyUsedOnce(Class<? extends Vertex> vertexClass) {
        List<Constructor> parentConstructor = getConstructorsWithAnnotatedParameters(vertexClass,
            LoadParentVertex.class);
        assertThat("Need Constructor for Class: " + vertexClass, parentConstructor.size(), is(1));
        Map<String, Class> requiredParameters = new HashMap<>();

        for (Parameter parameter : parentConstructor.get(0).getParameters()) {
            LoadParentVertex annotation = parameter.getAnnotation(LoadParentVertex.class);
            assertThat("Annotation has to be present on all params for class: " + vertexClass, annotation,
                is(notNullValue()));
            assertThat("Annotation can only be used once for class: " + vertexClass, requiredParameters,
                not(hasKey(annotation.value())));
            requiredParameters.put(annotation.value(), parameter.getType());
        }

        return requiredParameters;
    }
}
