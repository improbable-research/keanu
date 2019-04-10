package io.improbable.keanu.util.io;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Longs;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.templating.Sequence;
import io.improbable.keanu.templating.SequenceBuilder;
import io.improbable.keanu.templating.SequenceConstructionException;
import io.improbable.keanu.templating.SequenceItem;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.SimpleVertexDictionary;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;
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
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
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
        DoubleIfVertex outputVertex = (DoubleIfVertex) complexNet.getVertexByLabel(new VertexLabel(OUTPUT_NAME));
        GaussianVertex inputVertex = (GaussianVertex) complexNet.getVertexByLabel(new VertexLabel(INPUT_NAME));

        ByteArrayInputStream input = new ByteArrayInputStream(outputStream.toByteArray());
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork loadedNet = loader.loadNetwork(input);
        DoubleIfVertex outputVertex2 = (DoubleIfVertex) loadedNet.getVertexByLabel(new VertexLabel(OUTPUT_NAME));
        GaussianVertex inputVertex2 = (GaussianVertex) loadedNet.getVertexByLabel(new VertexLabel(INPUT_NAME));

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
        expectedException.expectMessage("Failed to create vertex due to missing parent: sigma");

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
        expectedException.expectMessage("Failed to create vertex due to missing parent: constant");

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

    private class TestNonSaveableVertex extends DoubleVertex implements NonSaveableVertex {

        private TestNonSaveableVertex() {
            super(new long[]{1, 1});
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
        Map<String, Class> requiredParams = checkConstructorParamValidityAndGetRequiredSaves(vertexClass);

        for (Map.Entry<String, Class> param : requiredParams.entrySet()) {
            assertThat("Class must save all required params: " + vertexClass,
                storedParams, hasKey(param.getKey()));
            if (param.getKey().equals("label")) {
                //Labels are stored as strings and are parsed into VertexLabels at load time
                assertThat(vertexClass + ": Saved and Loaded Param " + param.getKey() + " must have same type: "
                        + storedParams.get(param.getKey()) + ", " + param.getValue(),
                    String.class.isAssignableFrom(storedParams.get(param.getKey())));
            } else {
                assertThat(vertexClass + ": Saved and Loaded Param " + param.getKey() + " must have same type: "
                        + storedParams.get(param.getKey()) + ", " + param.getValue(),
                    param.getValue().isAssignableFrom(storedParams.get(param.getKey())));
            }
        }
    }

    private <A extends AnnotatedElement> List<A> filterAnnotatedObjects(A[] items, Class annotation) {
        List<A> filteredList = new ArrayList<>();

        Arrays.stream(items)
            .filter(item -> item.isAnnotationPresent(annotation))
            .forEach(item -> filteredList.add((A) item));

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

    private Map<String, Class> checkConstructorParamValidityAndGetRequiredSaves(Class<? extends Vertex> vertexClass) {
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

    @Test
    public void youCanConstructSingleSequenceItem() throws IOException {
        VertexLabel xLabel = new VertexLabel("x");

        DoubleVertex two = new ConstantDoubleVertex(2.0);

        Consumer<SequenceItem> factory = sequenceItem -> {
            DoubleProxyVertex xInput = sequenceItem.addDoubleProxyFor(xLabel);
            DoubleVertex xOutput = xInput.multiply(two).setLabel(xLabel);

            sequenceItem.add(xOutput);
        };

        DoubleVertex xInitial = new ConstantDoubleVertex(1.0).setLabel(xLabel);
        VertexDictionary initialState = SimpleVertexDictionary.of(xInitial);

        Sequence sequence = new SequenceBuilder()
            .withInitialState(initialState)
            .count(2)
            .withFactory(factory)
            .build();
        BayesianNetwork network = sequence.toBayesianNetwork();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reloadedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Sequence reconstructedSequence = Sequence.loadFromBayesNet(reloadedNetwork);

        assertThat(reconstructedSequence.size(), is(2));
        assertThat(reconstructedSequence.getUniqueIdentifier(), is(sequence.getUniqueIdentifier()));

        List<SequenceItem> originalList = sequence.asList();
        VertexLabel xProxyLabel = SequenceBuilder.proxyLabelFor(xLabel);
        reconstructedSequence.forEach(sequenceItem -> {
            SequenceItem originalItem = originalList.get(sequenceItem.getIndex());
            assertThat(sequenceItem.getContents().keySet(), is(originalItem.getContents().keySet()));
            assertThat(sequenceItem.get(xLabel), notNullValue());
            assertThat(sequenceItem.get(xProxyLabel), notNullValue());
        });

        Vertex<? extends DoubleTensor> outputVertex = reconstructedSequence.getLastItem().get(xLabel);
        double actualOutputValue = outputVertex.getValue().scalar();
        assertThat(actualOutputValue, is(4.0));
    }

    @Test
    public void youCanConstructASequenceItem() throws IOException {
        VertexLabel xLabel = new VertexLabel("x");
        Sequence sequence = constructSimpleSequence(null, xLabel);
        BayesianNetwork network = sequence.toBayesianNetwork();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Sequence reconstructedSequence = Sequence.loadFromBayesNet(reconstructedNetwork);

        assertThat(reconstructedSequence.size(), is(2));
        assertThat(reconstructedSequence.getUniqueIdentifier(), is(sequence.getUniqueIdentifier()));

        assertSequenceContains(sequence, reconstructedSequence, xLabel, SequenceBuilder.proxyLabelFor(xLabel));
    }

    @Test
    public void itThrowsWhenManyAreStoredButOneIsRequested() throws IOException {
        expectedException.expect(SequenceConstructionException.class);
        expectedException.expectMessage("The provided BayesianNetwork contains more than one Sequence");

        VertexLabel x1Label = new VertexLabel("x1");
        VertexLabel x2Label = new VertexLabel("x2");
        VertexLabel outputLabel = new VertexLabel("OUTPUT");
        String sequence1Label = "Sequence_1";
        String sequence2Label = "Sequence_2";

        Sequence sequence1 = constructSimpleSequence(sequence1Label, x1Label);
        Sequence sequence2 = constructSimpleSequence(sequence2Label, x2Label);

        DoubleVertex output1 = sequence1.getLastItem().get(x1Label);
        DoubleVertex output2 = sequence2.getLastItem().get(x2Label);

        DoubleVertex masterOutput = output1.multiply(output2).setLabel(outputLabel);

        BayesianNetwork network = new BayesianNetwork(masterOutput.getConnectedGraph());

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Sequence reconstructedSequence = Sequence.loadFromBayesNet(reconstructedNetwork);
    }

    @Test
    public void youCanConstructManySequenceItems() throws IOException {
        VertexLabel x1Label = new VertexLabel("x1");
        VertexLabel x2Label = new VertexLabel("x2");
        VertexLabel outputLabel = new VertexLabel("OUTPUT");
        String sequence1Label = "Sequence_1";
        String sequence2Label = "Sequence_2";

        Sequence sequence1 = constructSimpleSequence(sequence1Label, x1Label);
        Sequence sequence2 = constructSimpleSequence(sequence2Label, x2Label);

        DoubleVertex output1 = sequence1.getLastItem().get(x1Label);
        DoubleVertex output2 = sequence2.getLastItem().get(x2Label);

        DoubleVertex masterOutput = output1.multiply(output2).setLabel(outputLabel);

        BayesianNetwork network = new BayesianNetwork(masterOutput.getConnectedGraph());

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetworks = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Collection<Sequence> reconstructedSequences = Sequence.loadMultipleSequencesFromBayesNet(reconstructedNetworks);

        Sequence reconstructedSequence1 = getSequenceAndCheckNotNull(reconstructedSequences, sequence1Label);
        Sequence reconstructedSequence2 = getSequenceAndCheckNotNull(reconstructedSequences, sequence2Label);

        assertThat(reconstructedSequences.size(), is(2));
        assertThat(reconstructedSequence1.getUniqueIdentifier(), is(sequence1.getUniqueIdentifier()));
        assertThat(reconstructedSequence2.getUniqueIdentifier(), is(sequence2.getUniqueIdentifier()));
        assertThat(reconstructedSequence1.size(), is(2));
        assertThat(reconstructedSequence2.size(), is(2));
        assertThat(reconstructedSequence1.getName(), is(sequence1Label));
        assertThat(reconstructedSequence2.getName(), is(sequence2Label));

        VertexLabel x1ProxyLabel = SequenceBuilder.proxyLabelFor(x1Label);
        VertexLabel x2ProxyLabel = SequenceBuilder.proxyLabelFor(x2Label);
        assertSequenceContains(sequence1, reconstructedSequence1, x1Label, x1ProxyLabel);
        assertSequenceContains(sequence2, reconstructedSequence2, x2Label, x2ProxyLabel);

        BayesianNetwork reconstructedNetwork = sequence1.toBayesianNetwork();
        Vertex reconstructedMasterOutput = reconstructedNetwork.getVertexByLabel(outputLabel);
        assertThat(reconstructedMasterOutput, notNullValue());
        assertThat(((DoubleTensor) reconstructedMasterOutput.getValue()).scalar(), is(16.0));
    }

    private void assertSequenceContains(Sequence sequence, Sequence reconstructedSequence, VertexLabel xLabel, VertexLabel xProxyLabel) {
        List<SequenceItem> originalList = sequence.asList();
        reconstructedSequence.forEach(sequenceItem -> {
            SequenceItem originalItem = originalList.get(sequenceItem.getIndex());
            assertThat(sequenceItem.getContents().keySet(), is(originalItem.getContents().keySet()));
            assertThat(sequenceItem.get(xLabel), notNullValue());
            assertThat(sequenceItem.get(xProxyLabel), notNullValue());
        });

        Vertex<? extends DoubleTensor> outputVertex2 = reconstructedSequence.getLastItem().get(xLabel);
        double actualOutputValue2 = outputVertex2.getValue().scalar();
        assertThat(actualOutputValue2, is(4.0));
    }

    private Sequence getSequenceAndCheckNotNull(Collection<Sequence> sequences, String sequenceName) {
        Sequence reconstructedSequence = sequences
            .stream()
            .filter(sequence -> sequence.getName().equals(sequenceName))
            .findFirst()
            .orElse(null);
        assertThat(reconstructedSequence, notNullValue());
        return reconstructedSequence;
    }

    private Sequence constructSimpleSequence(String sequenceName, VertexLabel outputLabel) {
        DoubleVertex two = new ConstantDoubleVertex(2.0);

        Consumer<SequenceItem> factory = sequenceItem -> {
            DoubleProxyVertex xInput = sequenceItem.addDoubleProxyFor(outputLabel);
            DoubleVertex xOutput = xInput.multiply(two).setLabel(outputLabel);

            sequenceItem.add(xOutput);
        };

        DoubleVertex xInitial = new ConstantDoubleVertex(1.0).setLabel(outputLabel);
        VertexDictionary initialState = SimpleVertexDictionary.of(xInitial);

        SequenceBuilder builder = new SequenceBuilder();
        if (sequenceName != null) {
            builder = builder.named(sequenceName);
        }

        return builder
            .withInitialState(initialState)
            .count(2)
            .withFactory(factory)
            .build();
    }

}
