package io.improbable.keanu.network;

import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;
import org.reflections.Reflections;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.AnnotatedElement;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
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
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.hasKey;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertEquals;

public class ProtobufTest {

    private final static String GAUSS_LABEL = "GAUSSIAN VERTEX";
    private final static String GAUSS_ID = "1.1";
    private final static Double GAUSS_VALUE = 1.75;
    private final static String OUTPUT_NAME = "Output";
    private final static String INPUT_NAME = "Input";

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void youCanSaveAndLoadANetworkWithValues() throws IOException {
        final String gaussianLabel = "Gaussian";
        DoubleVertex mu1 = new ConstantDoubleVertex(new double[]{3.0, 1.0});
        DoubleVertex mu2 = new ConstantDoubleVertex(new double[]{5.0, 6.0});
        DoubleVertex finalMu = new ConcatenationVertex(0, mu1, mu2);
        DoubleVertex gaussianVertex = new GaussianVertex(finalMu, 1.0);
        gaussianVertex.setLabel(gaussianLabel);
        BayesianNetwork net = new BayesianNetwork(gaussianVertex.getConnectedGraph());
        ByteArrayOutputStream output = new ByteArrayOutputStream();

        ProtobufSaver protobufSaver = new ProtobufSaver(net);
        protobufSaver.save(output, true);
        assertThat(output.size(), greaterThan(0));
        ByteArrayInputStream input = new ByteArrayInputStream(output.toByteArray());

        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(input);

        assertThat(readNet.getLatentVertices().size(), is(1));
        assertThat(readNet.getLatentVertices().get(0), instanceOf(GaussianVertex.class));
        GaussianVertex latentGaussianVertex = (GaussianVertex)readNet.getLatentVertices().get(0);
        GaussianVertex labelGaussianVerted = (GaussianVertex)readNet.getVertexByLabel(new VertexLabel(gaussianLabel));
        assertThat(latentGaussianVertex, equalTo(labelGaussianVerted));
        assertThat(latentGaussianVertex.getMu().getValue(0), closeTo(3.0, 1e-10));
        assertThat(labelGaussianVerted.getMu().getValue(2), closeTo(5.0, 1e-10));
        assertThat(latentGaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        latentGaussianVertex.sample();
    }

    @Test
    public void shapeIsCorrectlySavedAndLoaded() throws IOException {
        DoubleVertex gaussianVertex1 = new GaussianVertex(new long[] {2, 3},0.0, 1.0);
        DoubleVertex gaussianVertex2 = new GaussianVertex(new long[] {3, 2}, 0.0, 1.0);
        DoubleVertex output = gaussianVertex1.matrixMultiply(gaussianVertex2);
        BayesianNetwork bayesNet = new BayesianNetwork(output.getConnectedGraph());

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(bayesNet);
        saver.save(outputStream, false);

        ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
        ProtobufLoader loader = new ProtobufLoader();

        BayesianNetwork readNet = loader.loadNetwork(inputStream);
    }

    @Test
    public void saveLoadGradientTest() throws IOException {
        BayesianNetwork complexNet = createComplexNet();
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(complexNet);
        saver.save(outputStream, true);
        DoubleIfVertex outputVertex = (DoubleIfVertex)complexNet.getVertexByLabel(new VertexLabel(OUTPUT_NAME));
        DoubleVertex inputVertex = (DoubleVertex)complexNet.getVertexByLabel(new VertexLabel(INPUT_NAME));

        ByteArrayInputStream input = new ByteArrayInputStream(outputStream.toByteArray());
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork loadedNet = loader.loadNetwork(input);
        DoubleIfVertex outputVertex2 = (DoubleIfVertex)loadedNet.getVertexByLabel(new VertexLabel(OUTPUT_NAME));
        DoubleVertex inputVertex2 = (DoubleVertex)loadedNet.getVertexByLabel(new VertexLabel(INPUT_NAME));

        DoubleTensor dOutputBefore = Differentiator.forwardModeAutoDiff(outputVertex).withRespectTo(inputVertex);
        DoubleTensor dOutputAfter = Differentiator.forwardModeAutoDiff(outputVertex2).withRespectTo(inputVertex2);

        assertEquals(dOutputBefore, dOutputAfter);

        dOutputBefore = Differentiator.reverseModeAutoDiff(outputVertex, inputVertex).withRespectTo(inputVertex);
        dOutputAfter = Differentiator.reverseModeAutoDiff(outputVertex2, inputVertex2).withRespectTo(inputVertex2);

        assertEquals(dOutputBefore, dOutputAfter);
    }

    private BayesianNetwork createComplexNet() {
        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1).setLabel(INPUT_NAME);
        A.setValue(DoubleTensor.create(3.0, new long[]{2, 2}));
        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 0, 1);
        B.setValue(DoubleTensor.create(5.0, new long[]{2, 2}));
        DoubleVertex D = A.times(B);
        DoubleVertex C = A.sin();
        DoubleVertex E = C.times(D);
        DoubleVertex G = E.log();
        DoubleVertex F = D.plus(B);

        BoolVertex predicate = ConstantVertex.of(BooleanTensor.create(new boolean[]{true, false, true, false}, new long[]{2, 2}));
        DoubleVertex H = If.isTrue(predicate).then(G).orElse(F).setLabel(OUTPUT_NAME);

        return new BayesianNetwork(H.getConnectedGraph());
    }

    @Test
    public void loadFailsIfParentsAreMissing() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex due to missing parent: sigma");

        KeanuSavedBayesNet.Vertex muVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setLabel("MU VERTEX")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(KeanuSavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1))
                    .addValues(0.0).build())
                .build())
            .build();

        KeanuSavedBayesNet.Vertex gaussianVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("2"))
            .setLabel("GAUSSIAN VERTEX")
            .setVertexType(GaussianVertex.class.getCanonicalName())
            .addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("mu")
                .setParentVertex(KeanuSavedBayesNet.VertexID.newBuilder().setId("1").build())
                .build()
            )
            .build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(muVertex)
            .addVertices(gaussianVertex).build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);

        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork net = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfInvalidVertexSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Unknown Vertex Type Specified: made.up.vertex.NonExistentVertex");

        KeanuSavedBayesNet.Vertex constantVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType("made.up.vertex.NonExistentVertex")
            .build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(constantVertex).build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfNoConstantSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex due to missing parent: constant");

        KeanuSavedBayesNet.Vertex constantVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(constantVertex).build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfWrongArgumentTypeSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Incorrect Parameter Type specified.  " +
            "Got: class io.improbable.keanu.tensor.intgr.ScalarIntegerTensor, " +
            "Expected: interface io.improbable.keanu.tensor.dbl.DoubleTensor");

        KeanuSavedBayesNet.Vertex constantVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setIntTensorParam(KeanuSavedBayesNet.IntegerTensor.newBuilder()
                    .addAllShape(Longs.asList()).addValues(1).build()
                ).build())
            .build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(constantVertex).build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void canLoadWithLabelRatherThanId() throws IOException {
        KeanuSavedBayesNet.BayesianNetwork savedNet = createBasicNetworkProtobufWithValue(
            GAUSS_LABEL, GAUSS_ID, GAUSS_VALUE);

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);

        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork net = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        GaussianVertex newGauss = (GaussianVertex)net.getVertexByLabel(new VertexLabel(GAUSS_LABEL));
        assertThat(newGauss.getValue().scalar(), is(GAUSS_VALUE));
    }

    @Test
    public void loadFailsWithConflictingVertexInfoInValue() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Label and VertexID don't refer to same Vertex: (sigmaVertex) " +
            "(id: \"1.1\"\n)");

        KeanuSavedBayesNet.BayesianNetwork savedNet = createBasicNetworkProtobufWithValue(
            "sigmaVertex", GAUSS_ID, 2.1
        );

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);

        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork net = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfValueIsWrongType() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Non Double Value specified for Double Vertex");

        KeanuSavedBayesNet.Vertex constantVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(KeanuSavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1)).addValues(1.0).build()
                ).build())
            .build();

        KeanuSavedBayesNet.StoredValue constantValue = KeanuSavedBayesNet.StoredValue.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setValue(KeanuSavedBayesNet.VertexValue.newBuilder()
                .setIntVal(KeanuSavedBayesNet.IntegerTensor.newBuilder()
                    .addShape(1).addShape(1)
                    .addValues(2)
                    .build()
                ).build()
            ).build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(constantVertex)
            .addDefaultState(constantValue)
            .build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedNet.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    private KeanuSavedBayesNet.BayesianNetwork createBasicNetworkProtobufWithValue(String labelForValue,
                                                                                   String idForValue,
                                                                                   Double valueToStore) {

        KeanuSavedBayesNet.Vertex muVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("1"))
            .setLabel("muVertex")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(KeanuSavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1)).addValues(1.0).build()
                ).build())
            .build();

        KeanuSavedBayesNet.Vertex sigmaVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId("2"))
            .setLabel("sigmaVertex")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(KeanuSavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1)).addValues(2.0).build()
                ).build())
            .build();

        KeanuSavedBayesNet.Vertex gaussianVertex = KeanuSavedBayesNet.Vertex.newBuilder()
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId(GAUSS_ID))
            .setLabel(GAUSS_LABEL)
            .setVertexType(GaussianVertex.class.getCanonicalName())
            .addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("mu")
                .setParentVertex(KeanuSavedBayesNet.VertexID.newBuilder().setId("1").build())
                .build()
            ).addParameters(KeanuSavedBayesNet.NamedParam.newBuilder()
                .setName("sigma")
                .setParentVertex(KeanuSavedBayesNet.VertexID.newBuilder().setId("2").build())
                .build()
            )
            .build();

        KeanuSavedBayesNet.StoredValue gaussianValue = KeanuSavedBayesNet.StoredValue.newBuilder()
            .setVertexLabel(labelForValue)
            .setId(KeanuSavedBayesNet.VertexID.newBuilder().setId(idForValue))
            .setValue(KeanuSavedBayesNet.VertexValue.newBuilder()
                .setDoubleVal(KeanuSavedBayesNet.DoubleTensor.newBuilder()
                    .addShape(1).addShape(1)
                    .addValues(valueToStore)
                    .build()
                ).build()
            ).build();

        KeanuSavedBayesNet.BayesianNetwork savedNet = KeanuSavedBayesNet.BayesianNetwork.newBuilder()
            .addVertices(muVertex)
            .addVertices(sigmaVertex)
            .addVertices(gaussianVertex)
            .addDefaultState(gaussianValue)
            .build();

        return savedNet;
    }

    private class TestNonSaveableVertex extends DoubleVertex implements NonSaveableVertex {

        private TestNonSaveableVertex() {
            super(new long[]{1, 1});
        }

        @Override
        public DoubleTensor sample(KeanuRandom random) {
            return null;
        }

        @Override
        public DoubleTensor sample() {
            return null;
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void nonSaveableVertexThrowsExceptionOnSave() {
        DoubleVertex testVertex = new TestNonSaveableVertex();
        BayesianNetwork net = new BayesianNetwork(testVertex.getConnectedGraph());
        ProtobufSaver protobufSaver = new ProtobufSaver(net);
        testVertex.save(protobufSaver);
    }

    @Category(Slow.class)
    @Test
    public void allSaveableVerticesHaveCorrectAnnotations() {
        Reflections reflections = new Reflections("io.improbable.keanu.vertices");

        Set<Class<? extends Vertex>> vertices = reflections.getSubTypesOf(Vertex.class);
        vertices.stream()
            .filter(v -> !NonSaveableVertex.class.isAssignableFrom(v))
            .filter(v -> !Modifier.isAbstract(v.getModifiers()))
            .forEach(this::checkSaveableVertex);
    }

    private void checkSaveableVertex(Class<? extends Vertex> vertexClass) {
        /*
         * For each vertex we need to check that we have a single constructor we can use for loading and that we save
         * all the necessary Params for that constructor
         */
        Map<String, Class> storedParams = getSavedParams(vertexClass);
        Map<String, Class> requiredParams = getRequiredParamsAndCheckOnlyUsedOnce(vertexClass);

        for (Map.Entry<String, Class> param : requiredParams.entrySet()) {
            assertThat("Class must save all required params: " + vertexClass,
                storedParams, hasKey(param.getKey()));
            assertThat(vertexClass + ": Saved and Loaded Param " + param.getKey() + " must have same type: "
                    + storedParams.get(param.getKey()) + ", " + param.getValue(),
                param.getValue().isAssignableFrom(storedParams.get(param.getKey())));
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

    private Map<String, Class> getSavedParams(Class<? extends Vertex> vertexClass) {
        Map<String, Class> savedParams = new HashMap<>();

        for (Method method : filterAnnotatedObjects(vertexClass.getMethods(), SaveVertexParam.class)) {
            String paramName = method.getAnnotation(SaveVertexParam.class).value();
            Class paramType = method.getReturnType();
            savedParams.put(paramName, paramType);
        }

        return savedParams;
    }

    private Map<String, Class> getRequiredParamsAndCheckOnlyUsedOnce(Class<? extends Vertex> vertexClass) {
        List<Constructor> parentConstructor = getConstructorsWithAnnotatedParameters(vertexClass,
            LoadVertexParam.class);
        assertThat("Need Constructor for Class: " + vertexClass, parentConstructor.size(), is(1));
        Map<String, Class> requiredParameters = new HashMap<>();

        for (Parameter parameter : parentConstructor.get(0).getParameters()) {
            LoadVertexParam parameterAnnotation = parameter.getAnnotation(LoadVertexParam.class);
            LoadShape shapeAnnotation = parameter.getAnnotation(LoadShape.class);
            assertThat("Annotation has to be present on all Constructor params for class: " + vertexClass,
                parameterAnnotation != null || shapeAnnotation != null);
            if (parameterAnnotation != null) {
                assertThat("Annotation can only be used once for class: " + vertexClass, requiredParameters,
                    not(hasKey(parameterAnnotation.value())));
                requiredParameters.put(parameterAnnotation.value(), parameter.getType());
            } else {
                assertThat("Shape Arguments must be long[]", parameter.getType().isAssignableFrom(long[].class));
            }
        }

        return requiredParameters;
    }
}
