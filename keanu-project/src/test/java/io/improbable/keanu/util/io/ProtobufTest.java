package io.improbable.keanu.util.io;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Longs;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.mir.KeanuSavedBayesNet;
import io.improbable.mir.SavedBayesNet;
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
import java.util.HashSet;
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
import static org.hamcrest.Matchers.notNullValue;
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
        GaussianVertex latentGaussianVertex = (GaussianVertex) readNet.getLatentVertices().get(0);
        GaussianVertex labelGaussianVertex = (GaussianVertex) readNet.getVertexByLabel(new VertexLabel(gaussianLabel));
        assertThat(latentGaussianVertex, equalTo(labelGaussianVertex));
        assertThat(latentGaussianVertex.getMu().getValue(0), closeTo(3.0, 1e-10));
        assertThat(labelGaussianVertex.getMu().getValue(2), closeTo(5.0, 1e-10));
        assertThat(latentGaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        latentGaussianVertex.sample();
    }

    @Test
    public void shapeIsCorrectlySavedAndLoaded() throws IOException {
        long[] shape1 = new long[]{2, 3};
        long[] shape2 = new long[]{3, 2};
        final VertexLabel LABEL_ONE = new VertexLabel("Vertex1");
        final VertexLabel LABEL_TWO = new VertexLabel("Vertex2");

        DoubleVertex gaussianVertex1 = new GaussianVertex(shape1, 0.0, 1.0);
        gaussianVertex1.setLabel(LABEL_ONE);
        DoubleVertex gaussianVertex2 = new GaussianVertex(shape2, 0.0, 1.0);
        gaussianVertex2.setLabel(LABEL_TWO);
        DoubleVertex output = gaussianVertex1.matrixMultiply(gaussianVertex2);
        BayesianNetwork bayesNet = new BayesianNetwork(output.getConnectedGraph());

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(bayesNet);
        saver.save(outputStream, false);

        ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
        ProtobufLoader loader = new ProtobufLoader();

        BayesianNetwork readNet = loader.loadNetwork(inputStream);
        Vertex vertexToShapeCheck = readNet.getVertexByLabel(LABEL_ONE);
        assertThat(vertexToShapeCheck.getShape(), is(shape1));
        vertexToShapeCheck = readNet.getVertexByLabel(LABEL_TWO);
        assertThat(vertexToShapeCheck.getShape(), is(shape2));
    }

    @Test
    public void saveLoadGradientTest() throws IOException {
        BayesianNetwork complexNet = createComplexNet();
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(complexNet);
        saver.save(outputStream, true);
        DoubleVertex outputVertex = (DoubleVertex) complexNet.getVertexByLabel(new VertexLabel(OUTPUT_NAME));
        DoubleVertex inputVertex = (DoubleVertex) complexNet.getVertexByLabel(new VertexLabel(INPUT_NAME));

        ByteArrayInputStream input = new ByteArrayInputStream(outputStream.toByteArray());
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork loadedNet = loader.loadNetwork(input);
        DoubleVertex outputVertex2 = (DoubleVertex) loadedNet.getVertexByLabel(new VertexLabel(OUTPUT_NAME));
        DoubleVertex inputVertex2 = (DoubleVertex) loadedNet.getVertexByLabel(new VertexLabel(INPUT_NAME));

        DoubleTensor dOutputBefore = Differentiator.forwardModeAutoDiff(
            inputVertex,
            outputVertex
        ).of(outputVertex);

        DoubleTensor dOutputAfter = Differentiator.forwardModeAutoDiff(
            inputVertex2,
            outputVertex2
        ).of(outputVertex2);

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

        BooleanVertex predicate = ConstantVertex.of(BooleanTensor.create(new boolean[]{true, false, true, false}, new long[]{2, 2}));
        DoubleVertex H = If.isTrue(predicate).then(G).orElse(F).setLabel(OUTPUT_NAME);

        return new BayesianNetwork(H.getConnectedGraph());
    }

    @Test
    public void loadFailsIfParentsAreMissing() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex due to missing parameter: sigma");

        SavedBayesNet.Vertex muVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("1"))
            .setLabel("MU VERTEX")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(SavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1))
                    .addValues(0.0).build())
                .build())
            .build();

        SavedBayesNet.Vertex gaussianVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("2"))
            .setLabel("GAUSSIAN VERTEX")
            .setVertexType(GaussianVertex.class.getCanonicalName())
            .addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("mu")
                .setParentVertex(SavedBayesNet.VertexID.newBuilder().setId("1").build())
                .build()
            )
            .build();

        SavedBayesNet.Graph savedNet = SavedBayesNet.Graph.newBuilder()
            .addVertices(muVertex)
            .addVertices(gaussianVertex).build();

        KeanuSavedBayesNet.ProtoModel savedModel = KeanuSavedBayesNet.ProtoModel.newBuilder()
            .setGraph(savedNet)
            .build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedModel.writeTo(writer);

        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork net = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfInvalidVertexSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Unknown Vertex Type Specified: made.up.vertex.NonExistentVertex");

        SavedBayesNet.Vertex constantVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType("made.up.vertex.NonExistentVertex")
            .build();

        SavedBayesNet.Graph savedNet = SavedBayesNet.Graph.newBuilder()
            .addVertices(constantVertex).build();

        KeanuSavedBayesNet.ProtoModel savedModel = KeanuSavedBayesNet.ProtoModel.newBuilder()
            .setGraph(savedNet)
            .build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedModel.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfNoConstantSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex due to missing parameter: constant");

        SavedBayesNet.Vertex constantVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .build();

        SavedBayesNet.Graph savedNet = SavedBayesNet.Graph.newBuilder()
            .addVertices(constantVertex).build();

        KeanuSavedBayesNet.ProtoModel savedModel = KeanuSavedBayesNet.ProtoModel.newBuilder()
            .setGraph(savedNet)
            .build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedModel.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfWrongArgumentTypeSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Incorrect Parameter Type specified.  " +
            "Got: class io.improbable.keanu.tensor.intgr.ScalarIntegerTensor, " +
            "Expected: interface io.improbable.keanu.tensor.dbl.DoubleTensor");

        SavedBayesNet.Vertex constantVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setIntTensorParam(SavedBayesNet.IntegerTensor.newBuilder()
                    .addAllShape(Longs.asList()).addValues(1).build()
                ).build())
            .build();

        SavedBayesNet.Graph savedNet = SavedBayesNet.Graph.newBuilder()
            .addVertices(constantVertex).build();

        KeanuSavedBayesNet.ProtoModel savedModel = KeanuSavedBayesNet.ProtoModel.newBuilder()
            .setGraph(savedNet)
            .build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedModel.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void canLoadWithLabelRatherThanId() throws IOException {
        KeanuSavedBayesNet.ProtoModel savedModel =
            createBasicNetworkProtobufWithValue(GAUSS_LABEL, GAUSS_ID, GAUSS_VALUE);

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedModel.writeTo(writer);

        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork net = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        GaussianVertex newGauss = (GaussianVertex) net.getVertexByLabel(new VertexLabel(GAUSS_LABEL));
        assertThat(newGauss.getValue().scalar(), is(GAUSS_VALUE));
    }

    @Test
    public void metadataCanBeSavedToProtobuf() throws IOException {
        Vertex vertex = new ConstantIntegerVertex(1);
        BayesianNetwork net = new BayesianNetwork(vertex.getConnectedGraph());
        Map<String, String> metadata = ImmutableMap.of("Author", "Some Author", "Tag", "MyBayesNet");

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver protobufSaver = new ProtobufSaver(net);
        protobufSaver.save(writer, true, metadata);
        KeanuSavedBayesNet.ProtoModel parsedModel = KeanuSavedBayesNet.ProtoModel.parseFrom(writer.toByteArray());

        KeanuSavedBayesNet.ModelMetadata.Builder metadataBuilder = KeanuSavedBayesNet.ModelMetadata.newBuilder();
        String[] metadataKeys = metadata.keySet().toArray(new String[0]);
        Arrays.sort(metadataKeys);
        for (String metadataKey : metadataKeys) {
            metadataBuilder.putMetadataInfo(metadataKey, metadata.get(metadataKey));
        }

        assertEquals(parsedModel.getMetadata().getMetadataInfoMap(), metadataBuilder.getMetadataInfoMap());
    }

    @Test
    public void loadFailsWithConflictingVertexInfoInValue() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Label and VertexID don't refer to same Vertex: (sigmaVertex) " +
            "(id: \"1.1\"\n)");

        KeanuSavedBayesNet.ProtoModel savedModel = createBasicNetworkProtobufWithValue(
            "sigmaVertex", GAUSS_ID, 2.1
        );

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedModel.writeTo(writer);

        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork net = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    @Test
    public void loadFailsIfValueIsWrongType() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Non Double Value specified for Double Vertex");

        SavedBayesNet.Vertex constantVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("1"))
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(SavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1)).addValues(1.0).build()
                ).build())
            .build();

        SavedBayesNet.StoredValue constantValue = SavedBayesNet.StoredValue.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("1"))
            .setValue(SavedBayesNet.VertexValue.newBuilder()
                .setIntVal(SavedBayesNet.IntegerTensor.newBuilder()
                    .addShape(1).addShape(1)
                    .addValues(2)
                    .build()
                ).build()
            ).build();

        SavedBayesNet.Graph savedNet = SavedBayesNet.Graph.newBuilder()
            .addVertices(constantVertex)
            .addDefaultState(constantValue)
            .build();

        KeanuSavedBayesNet.ProtoModel savedModel = KeanuSavedBayesNet.ProtoModel.newBuilder()
            .setGraph(savedNet)
            .build();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        savedModel.writeTo(writer);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork readNet = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
    }

    private KeanuSavedBayesNet.ProtoModel createBasicNetworkProtobufWithValue(String labelForValue,
                                                                              String idForValue,
                                                                              Double valueToStore) {

        SavedBayesNet.Vertex muVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("1"))
            .setLabel("muVertex")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(SavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1)).addValues(1.0).build()
                ).build())
            .build();

        SavedBayesNet.Vertex sigmaVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId("2"))
            .setLabel("sigmaVertex")
            .setVertexType(ConstantDoubleVertex.class.getCanonicalName())
            .addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("constant")
                .setDoubleTensorParam(SavedBayesNet.DoubleTensor.newBuilder()
                    .addAllShape(Longs.asList(1, 1)).addValues(2.0).build()
                ).build())
            .build();

        SavedBayesNet.Vertex gaussianVertex = SavedBayesNet.Vertex.newBuilder()
            .setId(SavedBayesNet.VertexID.newBuilder().setId(GAUSS_ID))
            .setLabel(GAUSS_LABEL)
            .setVertexType(GaussianVertex.class.getCanonicalName())
            .addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("mu")
                .setParentVertex(SavedBayesNet.VertexID.newBuilder().setId("1").build())
                .build()
            ).addParameters(SavedBayesNet.NamedParam.newBuilder()
                .setName("sigma")
                .setParentVertex(SavedBayesNet.VertexID.newBuilder().setId("2").build())
                .build()
            )
            .build();

        SavedBayesNet.StoredValue gaussianValue = SavedBayesNet.StoredValue.newBuilder()
            .setVertexLabel(labelForValue)
            .setId(SavedBayesNet.VertexID.newBuilder().setId(idForValue))
            .setValue(SavedBayesNet.VertexValue.newBuilder()
                .setDoubleVal(SavedBayesNet.DoubleTensor.newBuilder()
                    .addShape(1).addShape(1)
                    .addValues(valueToStore)
                    .build()
                ).build()
            ).build();

        SavedBayesNet.Graph savedNet = SavedBayesNet.Graph.newBuilder()
            .addVertices(muVertex)
            .addVertices(sigmaVertex)
            .addVertices(gaussianVertex)
            .addDefaultState(gaussianValue)
            .build();

        KeanuSavedBayesNet.ProtoModel savedModel = KeanuSavedBayesNet.ProtoModel.newBuilder()
            .setGraph(savedNet)
            .build();

        return savedModel;
    }

    private class TestNonSaveableVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, NonSaveableVertex {

        private TestNonSaveableVertex() {
            super(new long[]{1, 1});
        }

    }

    @Test(expected = IllegalArgumentException.class)
    public void nonSaveableVertexThrowsExceptionOnSave() {
        DoubleVertex testVertex = new TestNonSaveableVertex();
        BayesianNetwork net = new BayesianNetwork(testVertex.getConnectedGraph());
        ProtobufSaver protobufSaver = new ProtobufSaver(net);
        protobufSaver.save(testVertex);
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
        Map<String, Set<Class>> storedParams = getSavedParams(vertexClass);
        Map<String, Class> requiredParams = checkConstructorParamValidityAndGetRequiredSaves(vertexClass);

        for (Map.Entry<String, Class> param : requiredParams.entrySet()) {

            assertThat("Class must save all required params: " + vertexClass,
                storedParams, hasKey(param.getKey()));
            if (param.getKey().equals("label")) {
                //Labels are stored as strings and are parsed into VertexLabels at load time
                assertThat(vertexClass + ": Saved and Loaded Param " + param.getKey() + " must have same type: "
                        + storedParams.get(param.getKey()) + ", " + param.getValue(),
                    String.class.isAssignableFrom(storedParams.get(param.getKey()).iterator().next()));
            } else {
                assertThat(vertexClass + ": Saved and Loaded Param " + param.getKey() + " must have same type: "
                        + storedParams.get(param.getKey()) + ", " + param.getValue(),
                    containsAssignableFrom(storedParams.get(param.getKey()), param.getValue()));
            }
        }
    }

    private boolean containsAssignableFrom(Set<Class> paramTypes, Class toAssign) {
        return paramTypes.stream()
            .map(toAssign::isAssignableFrom)
            .collect(Collectors.toSet())
            .contains(true);
    }

    private <A extends AnnotatedElement> List<A> filterAnnotatedObjects(A[] items, Class annotation) {
        List<A> filteredList = new ArrayList<>();

        Arrays.stream(items)
            .filter(item -> item.isAnnotationPresent(annotation))
            .forEach(item -> filteredList.add((A) item));

        return filteredList;
    }

    private Set<Constructor> getConstructorsWithAnnotatedParameters(Class parentClass, Class annotation) {
        return Arrays.stream(parentClass.getConstructors())
            .filter(constructor -> !filterAnnotatedObjects(constructor.getParameters(), annotation).isEmpty())
            .collect(Collectors.toSet());
    }

    private Map<String, Set<Class>> getSavedParams(Class<? extends Vertex> vertexClass) {
        Map<String, Set<Class>> savedParams = new HashMap<>();

        for (Method method : filterAnnotatedObjects(vertexClass.getMethods(), SaveVertexParam.class)) {
            String paramName = method.getAnnotation(SaveVertexParam.class).value();
            Class paramType = method.getReturnType();
            savedParams.computeIfAbsent(paramName, (name) -> new HashSet<>()).add(paramType);
        }

        return savedParams;
    }

    private Map<String, Class> checkConstructorParamValidityAndGetRequiredSaves(Class<? extends Vertex> vertexClass) {
        Set<Constructor> loadConstructors = getConstructorsWithAnnotatedParameters(vertexClass, LoadVertexParam.class);
        loadConstructors.addAll(getConstructorsWithAnnotatedParameters(vertexClass, LoadShape.class));

        assertThat("Need Constructor for Class: " + vertexClass, loadConstructors.size(), is(1));
        Map<String, Class> requiredParameters = new HashMap<>();

        for (Parameter parameter : loadConstructors.iterator().next().getParameters()) {
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

    @Test
    public void canSaveAndLoadBayesNetWithUnparentedProxyVertex() throws IOException {
        VertexLabel proxyLabel = new VertexLabel("proxy");
        DoubleVertex proxyVertex = new DoubleProxyVertex(proxyLabel);

        BayesianNetwork network = new BayesianNetwork(proxyVertex.getConnectedGraph());

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Vertex reconstructedVertex = reconstructedNetwork.getVertexByLabel(proxyLabel);
        assertThat(reconstructedVertex, notNullValue());
    }
}
