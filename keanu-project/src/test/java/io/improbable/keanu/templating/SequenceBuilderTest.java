package io.improbable.keanu.templating;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.SimpleVertexDictionary;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexMatchers;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanProxyVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;

import static io.improbable.keanu.tensor.TensorMatchers.isScalarWithValue;
import static io.improbable.keanu.vertices.VertexLabelMatchers.hasUnqualifiedName;
import static io.improbable.keanu.vertices.VertexMatchers.hasLabel;
import static io.improbable.keanu.vertices.VertexMatchers.hasNoLabel;
import static io.improbable.keanu.vertices.VertexMatchers.hasParents;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class SequenceBuilderTest {

    private static class Bean {
        public int x;

        public Bean(int x) {
            this.x = x;
        }
    }

    private static final List<Bean> ROWS = Arrays.asList(
        new Bean(0),
        new Bean(0),
        new Bean(0)
    );

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void buildSequenceFromCount_Size() {
        int n = 100;
        Sequence sequence = new SequenceBuilder()
            .count(n)
            .withFactory(item -> {
            })
            .build();
        assertEquals(n, sequence.size());
    }

    @Test
    public void buildSequenceFromCount_Contents() {
        int n = 100;
        VertexLabel vertexName = new VertexLabel("vertexName");
        Sequence sequence = new SequenceBuilder<>()
            .count(n)
            .withFactory((item) -> item.add(vertexName, new BernoulliVertex(0.5)))
            .build();
        sequence.asList().forEach(item -> {
            assertNotNull(item.get(vertexName));
        });
    }

    @Test
    public void buildSequenceFromData_Size() {
        Sequence sequence = new SequenceBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((item, bean) -> {
            })
            .build();
        assertEquals(ROWS.size(), sequence.size());
    }

    @Test
    public void buildSequenceFromData_Contents() {
        Sequence sequence = new SequenceBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((item, bean) -> {
                assertEquals(0, bean.x);
            })
            .build();
    }

    @Test
    public void ifAVertexIsLabeledThatIsWhatsUsedToReferToItInTheSequenceItem() {
        VertexLabel label = new VertexLabel("label");

        Vertex<?> startVertex = ConstantVertex.of(1.).setLabel(label);

        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(startVertex)
            .withTransitionMapping(ImmutableMap.of(label, label))
            .count(10)
            .withFactory((item) -> {
                DoubleVertex intermediateVertex = new DoubleProxyVertex(label);
                item.add(intermediateVertex);
            })
            .build();

        for (SequenceItem item : sequence) {
            Vertex<?> vertex = item.get(label);
            assertThat(vertex, hasLabel(hasUnqualifiedName(label.getUnqualifiedName())));
            Vertex<?> parent = Iterables.getOnlyElement(vertex.getParents());
            assertThat(parent, hasLabel(hasUnqualifiedName(label.getUnqualifiedName())));
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void youCannotAddTheSameLabelTwiceIntoOneSequenceItem() {
        new SequenceBuilder<Integer>()
            .count(10)
            .withFactory((item) -> {
                VertexLabel label = new VertexLabel("x");
                DoubleVertex vertex1 = ConstantVertex.of(1.);
                DoubleVertex vertex2 = ConstantVertex.of(1.);
                item.add(label, vertex1);
                item.add(label, vertex2);
            })
            .build();
    }

    @Test
    public void youCanCreateASequenceWithACommonParameterFromACount() {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Sequence sequence = new SequenceBuilder<Bean>()
            .count(10)
            .withFactory((item) -> {
                BooleanVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                item.add(label, flip);
            })
            .build();


        for (SequenceItem item : sequence) {
            Vertex<DoubleTensor> flip = item.get(label);
            assertThat(flip.getParents(), contains(commonTheta));
        }
    }

    @Test
    public void youCanPutTheSequenceIntoABayesNet() {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Sequence sequence = new SequenceBuilder<Bean>()
            .count(10)
            .withFactory((item) -> {
                BooleanVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                item.add(label, flip);
            })
            .build();

        new BayesianNetwork(commonTheta.getConnectedGraph());
    }

    @Test
    public void youCanPutTheSameVertexIntoMultipleSequenceItems() {
        VertexLabel thetaLabel = new VertexLabel("theta");
        VertexLabel flipLabel = new VertexLabel("flip");
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        new SequenceBuilder<Bean>()
            .count(10)
            .withFactory((item) -> {
                BooleanVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                item.add(thetaLabel, commonTheta);
                item.add(flipLabel, flip);
            })
            .build();

        new BayesianNetwork(commonTheta.getConnectedGraph());
    }


    @Test
    public void youCanCreateASequenceWithACommonParameterFromAnIterator() {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Sequence sequence = new SequenceBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((item, bean) -> {
                BooleanVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                item.add(label, flip);
            })
            .build();


        for (SequenceItem item : sequence) {
            Vertex<DoubleTensor> flip = item.get(label);
            assertThat(flip.getParents(), contains(commonTheta));
        }
    }

    /**
     * This is a Hidden Markov Model -
     * see for example http://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf
     *
     * ...  -->  X[t-1]  -->  X[t]  --> ...
     *             |           |
     *           Y[t-1]       Y[t]
     */
    @Test
    public void youCanCreateATimeSeriesFromSequenceFromACount() {

        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xPreviousLabel = SequenceBuilder.proxyFor(xLabel);
        VertexLabel yLabel = new VertexLabel("y");

        Vertex<DoubleTensor> initialX = ConstantVertex.of(1.);
        List<Integer> ys = ImmutableList.of(0, 1, 2, 1, 3, 2);

        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(xLabel, initialX)
            .count(10)
            .withFactory((item) -> {
                DoubleVertex xPrevious = new DoubleProxyVertex(xPreviousLabel);
                DoubleVertex x = new ExponentialVertex(xPrevious);
                IntegerVertex y = new PoissonVertex(x);
                item.add(xPrevious);
                item.add(xLabel, x);
                item.add(yLabel, y);
            })
            .build();


        Vertex<DoubleTensor> previousX = initialX;

        for (SequenceItem item : sequence) {
            Vertex<DoubleTensor> xPreviousProxy = item.get(xPreviousLabel);
            Vertex<DoubleTensor> x = item.get(xLabel);
            Vertex<DoubleTensor> y = item.get(yLabel);
            assertThat(xPreviousProxy.getParents(), contains(previousX));
            assertThat(x.getParents(), contains(xPreviousProxy));
            assertThat(y.getParents(), contains(x));
            previousX = x;
        }
    }

    /**
     * This is a Hidden Markov Model -
     * see for example http://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf
     *
     * ...  -->  X[t-1]  -->  X[t]  --> ...
     *             |           |
     *           Y[t-1]       Y[t]
     */
    @Test
    public void youCanCreateATimeSeriesFromSequenceFromAnIterator() {

        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xPreviousLabel = SequenceBuilder.proxyFor(xLabel);
        VertexLabel yLabel = new VertexLabel("y");

        Vertex<DoubleTensor> initialX = ConstantVertex.of(1.);
        List<Integer> ys = ImmutableList.of(0, 1, 2, 1, 3, 2);

        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(xLabel, initialX)
            .fromIterator(ys.iterator())
            .withFactory((item, observedY) -> {
                DoubleVertex xPreviousProxy = new DoubleProxyVertex(xPreviousLabel);
                DoubleVertex x = new ExponentialVertex(xPreviousProxy);
                IntegerVertex y = new PoissonVertex(x);
                y.observe(observedY);
                item.add(xPreviousProxy);
                item.add(xLabel, x);
                item.add(yLabel, y);
            })
            .build();


        Vertex<DoubleTensor> previousX = initialX;

        int i = 0;
        for (SequenceItem item : sequence) {
            Vertex<DoubleTensor> xPreviousProxy = item.get(xPreviousLabel);
            Vertex<DoubleTensor> x = item.get(xLabel);
            Vertex<DoubleTensor> y = item.get(yLabel);
            assertThat(xPreviousProxy.getParents(), contains(previousX));
            assertThat(x.getParents(), contains(xPreviousProxy));
            assertThat(y.getParents(), contains(x));
            Optional<String> desiredLabelOuterNamespace = Optional.of("Sequence_Item_" + i);
            assertThat(x.getLabel().getOuterNamespace(), is(desiredLabelOuterNamespace));
            assertThat(y.getLabel().getOuterNamespace(), is(desiredLabelOuterNamespace));
            previousX = x;
            i++;
        }
    }

    /**
     * Note that this behaviour is wrapped by the Loop class
     * See LoopTest.java for example usage
     */
    @Test
    public void youCanCreateALoopFromSequenceFromACount() {
        // inputs
        VertexLabel runningTotalLabel = new VertexLabel("runningTotal");
        VertexLabel stillLoopingLabel = new VertexLabel("stillLooping");
        VertexLabel valueInLabel = new VertexLabel("valueIn");

        // intermediate
        VertexLabel oneLabel = new VertexLabel("one");
        VertexLabel conditionLabel = new VertexLabel("condition");

        // outputs
        VertexLabel plusLabel = new VertexLabel("plus");
        VertexLabel loopLabel = new VertexLabel("loop");
        VertexLabel valueOutLabel = new VertexLabel("valueOut");

        // base case
        DoubleVertex initialSum = ConstantVertex.of(0.);
        BooleanVertex tru = ConstantVertex.of(true);
        DoubleVertex initialValue = ConstantVertex.of(0.);

        int maximumLoopLength = 100;

        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(SimpleVertexDictionary.backedBy(ImmutableMap.of(
                plusLabel, initialSum,
                loopLabel, tru,
                valueOutLabel, initialValue)))
            .withTransitionMapping(ImmutableMap.of(
                runningTotalLabel, plusLabel,
                stillLoopingLabel, loopLabel,
                valueInLabel, valueOutLabel
            ))
            .count(maximumLoopLength)
            .withFactory((item) -> {
                // inputs
                DoubleVertex runningTotal = new DoubleProxyVertex(runningTotalLabel);
                BooleanVertex stillLooping = new BooleanProxyVertex(stillLoopingLabel);
                DoubleVertex valueIn = new DoubleProxyVertex(valueInLabel);
                item.addAll(ImmutableSet.of(runningTotal, stillLooping, valueIn));

                // intermediate
                DoubleVertex one = ConstantVertex.of(1.);
                BooleanVertex condition = new BernoulliVertex(0.5);
                item.add(oneLabel, one);
                item.add(conditionLabel, condition);

                // outputs
                DoubleVertex plus = runningTotal.plus(one);
                BooleanVertex loopAgain = stillLooping.and(condition);
                DoubleVertex result = If.isTrue(loopAgain).then(plus).orElse(valueIn);
                item.add(plusLabel, plus);
                item.add(loopLabel, loopAgain);
                item.add(valueOutLabel, result);
            })
            .build();


        DoubleVertex previousPlus = initialSum;
        BooleanVertex previousLoop = tru;
        DoubleVertex previousValueOut = initialValue;

        for (SequenceItem item : sequence) {
            DoubleVertex runningTotal = item.get(runningTotalLabel);
            BooleanVertex stillLooping = item.get(stillLoopingLabel);
            DoubleVertex valueIn = item.get(valueInLabel);

            DoubleVertex one = item.get(oneLabel);
            BooleanVertex condition = item.get(conditionLabel);

            DoubleVertex plus = item.get(plusLabel);
            BooleanVertex loop = item.get(loopLabel);
            DoubleVertex valueOut = item.get(valueOutLabel);

            assertThat(runningTotal.getParents(), contains(previousPlus));
            assertThat(stillLooping.getParents(), contains(previousLoop));
            assertThat(valueIn.getParents(), contains(previousValueOut));

            assertThat(one.getParents(), is(empty()));
            assertThat(condition, hasParents(contains(allOf(
                hasNoLabel(),
                instanceOf(ConstantDoubleVertex.class)
            ))));

            assertThat(plus.getParents(), containsInAnyOrder(runningTotal, one));
            assertThat(loop.getParents(), containsInAnyOrder(condition, stillLooping));
            assertThat(valueOut.getParents(), containsInAnyOrder(loop, valueIn, plus));

            previousPlus = plus;
            previousLoop = loop;
            previousValueOut = valueOut;
        }


        DoubleVertex output = sequence.asList().get(maximumLoopLength - 1).get(valueOutLabel);

        for (int firstFailure : new int[]{0, 1, 2, 10, 99}) {
            for (SequenceItem item : sequence) {
                BooleanVertex condition = item.get(conditionLabel);
                condition.setAndCascade(true);
            }
            BooleanVertex condition = sequence.asList().get(firstFailure).get(conditionLabel);
            condition.setAndCascade(false);
            Double expectedOutput = new Double(firstFailure);
            assertThat(output, VertexMatchers.hasValue(expectedOutput));
        }
    }

    @Test
    public void itThrowsIfTheresAProxyVertexThatItDoesntKnowHowToMap() {
        expectedException.expect(SequenceConstructionException.class);
        expectedException.expectMessage(startsWith("Cannot find transition mapping for "));
        VertexLabel realLabel = new VertexLabel("real");
        VertexLabel fakeLabel = new VertexLabel("fake");
        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(SimpleVertexDictionary.of())
            .withTransitionMapping(ImmutableMap.of(realLabel, realLabel))
            .count(10)
            .withFactory((item) -> {
                item.add(new DoubleProxyVertex(fakeLabel));
            })
            .build();
    }

    @Test
    public void itThrowsIfTheresAProxyVertexButNoBaseCase() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("You must provide a base case for the Transition Vertices - use withInitialState()");
        VertexLabel realLabel = new VertexLabel("real");
        Sequence sequence = new SequenceBuilder<Integer>()
            .withTransitionMapping(ImmutableMap.of(realLabel, realLabel))
            .count(10)
            .withFactory((item) -> {
                item.add(new DoubleProxyVertex(realLabel));
            })
            .build();
    }

    @Test
    public void itThrowsIfTheresAnUnknownLabelInTheProxyMapping() {
        expectedException.expect(SequenceConstructionException.class);
        expectedException.expectMessage("Cannot find VertexLabel fake");
        VertexLabel realLabel = new VertexLabel("real");
        VertexLabel fakeLabel = new VertexLabel("fake");
        DoubleVertex initialState = ConstantVertex.of(1.);
        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(realLabel, initialState)
            .withTransitionMapping(ImmutableMap.of(realLabel, fakeLabel))
            .count(10)
            .withFactory((item) -> {
                item.add(new DoubleProxyVertex(realLabel));
            })
            .build();
    }

    @Test
    public void testStillWorksWithNamespaceIdentifier() {
        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xPreviousLabel = SequenceBuilder.proxyFor(xLabel);
        VertexLabel yLabel = new VertexLabel("y");
        String uniqueNamespace = "UNIQUE";

        Vertex<DoubleTensor> initialX = ConstantVertex.of(1.);
        List<Integer> ys = ImmutableList.of(0, 1, 2, 1, 3, 2);

        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(xLabel, initialX)
            .named(uniqueNamespace)
            .fromIterator(ys.iterator())
            .withFactory((item, observedY) -> {
                DoubleVertex xPreviousProxy = new DoubleProxyVertex(xPreviousLabel);
                DoubleVertex x = new ExponentialVertex(xPreviousProxy);
                IntegerVertex y = new PoissonVertex(x);
                y.observe(observedY);
                item.add(xPreviousProxy);
                item.add(xLabel, x);
                item.add(yLabel, y);
            })
            .build();


        Vertex<DoubleTensor> previousX = initialX;

        for (SequenceItem item : sequence) {
            Vertex<DoubleTensor> xPreviousProxy = item.get(xPreviousLabel);
            Vertex<DoubleTensor> x = item.get(xLabel);
            Vertex<DoubleTensor> y = item.get(yLabel);
            assertThat(xPreviousProxy.getParents(), contains(previousX));
            assertThat(x.getParents(), contains(xPreviousProxy));
            assertThat(y.getParents(), contains(x));
            assertThat(x.getLabel().getOuterNamespace(), is(Optional.of(uniqueNamespace)));
            assertThat(y.getLabel().getOuterNamespace(), is(Optional.of(uniqueNamespace)));
            previousX = x;
        }
    }

    @Test
    public void youCanCreateSequenceFromMultipleFactories() {
        VertexLabel x1Label = new VertexLabel("x1");
        VertexLabel x2Label = new VertexLabel("x2");
        VertexLabel x3Label = new VertexLabel("x3");
        VertexLabel x4Label = new VertexLabel("x4");

        DoubleVertex two = new ConstantDoubleVertex(2);
        DoubleVertex half = new ConstantDoubleVertex(0.5);

        VertexLabel x1InputLabel = SequenceBuilder.proxyFor(x1Label);
        VertexLabel x2InputLabel = SequenceBuilder.proxyFor(x2Label);
        VertexLabel x3InputLabel = SequenceBuilder.proxyFor(x3Label);
        VertexLabel x4InputLabel = SequenceBuilder.proxyFor(x4Label);

        Consumer<SequenceItem> factory1 = sequenceItem -> {
            DoubleProxyVertex x1Input = new DoubleProxyVertex(x1InputLabel);
            DoubleProxyVertex x2Input = new DoubleProxyVertex(x2InputLabel);

            DoubleVertex x1Output = x1Input.multiply(two).setLabel(x1Label);
            DoubleVertex x3Output = x2Input.multiply(two).setLabel(x3Label);

            sequenceItem.addAll(x1Input, x2Input, x1Output, x3Output);
        };

        Consumer<SequenceItem> factory2 = sequenceItem -> {
            DoubleProxyVertex x3Input = new DoubleProxyVertex(x3InputLabel);
            DoubleProxyVertex x4Input = new DoubleProxyVertex(x4InputLabel);

            DoubleVertex x2Output = x3Input.multiply(half).setLabel(x2Label);
            DoubleVertex x4Output = x4Input.multiply(half).setLabel(x4Label);

            sequenceItem.addAll(x3Input, x4Input, x2Output, x4Output);
        };

        DoubleVertex x1Start = new ConstantDoubleVertex(4).setLabel(x1Label);
        DoubleVertex x2Start = new ConstantDoubleVertex(4).setLabel(x2Label);
        DoubleVertex x3Start = new ConstantDoubleVertex(4).setLabel(x3Label);
        DoubleVertex x4Start = new ConstantDoubleVertex(4).setLabel(x4Label);

        VertexDictionary dictionary = SimpleVertexDictionary.of(x1Start, x2Start, x3Start, x4Start);
        List<Consumer<SequenceItem>> factories = new ArrayList<>();
        factories.add(factory1);
        factories.add(factory2);


        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(dictionary)
            .count(5)
            .withFactories(factories)
            .build();

        assertThat(sequence.size(), equalTo(5));
        for (SequenceItem item : sequence.asList()) {
            checkSequenceOutputLinksToInput(item, x1InputLabel, x1Label);
            checkSequenceOutputLinksToInput(item, x2InputLabel, x3Label);
            checkSequenceOutputLinksToInput(item, x3InputLabel, x2Label);
            checkSequenceOutputLinksToInput(item, x4InputLabel, x4Label);
        }

        checkOutputEquals(sequence, x1Label, 128.);
        checkOutputEquals(sequence, x2Label, 2.);
        checkOutputEquals(sequence, x3Label, 8.);
        checkOutputEquals(sequence, x4Label, .125);
    }

    private void checkOutputEquals(Sequence sequence, VertexLabel label, Double desiredOutput) {
        DoubleVertex result = sequence.getLastItem().get(label);
        assertThat(result.getValue(), isScalarWithValue(desiredOutput));
    }

    /**
     * A helper method that finds the proxy vertex that is the output of the previous sequence item and checks it has been
     * wired up correctly to the desired input vertex of the current sequence item
     * desired
     */
    private void checkSequenceOutputLinksToInput(SequenceItem item, VertexLabel previousOutputLabel, VertexLabel currentInputLabel) {
        Set<Vertex> inputChildren = item.get(previousOutputLabel).getChildren();
        assertThat(inputChildren.size(), is(1));

        Optional<Vertex> optionalOutputOfPreviousTimestep = inputChildren.stream().findFirst();
        assertThat(optionalOutputOfPreviousTimestep.isPresent(), is(true));
        assertThat(optionalOutputOfPreviousTimestep.get().getLabel().withoutOuterNamespace().withoutOuterNamespace(), is(currentInputLabel));
    }

    @Test
    public void testYouCanNameANamespace() {
        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xInputLabel = SequenceBuilder.proxyFor(xLabel);

        DoubleVertex two = new ConstantDoubleVertex(2.0);

        Consumer<SequenceItem> factory = sequenceItem -> {
            DoubleProxyVertex xInput = new DoubleProxyVertex(xInputLabel);
            DoubleVertex xOutput = xInput.multiply(two).setLabel(xLabel);

            sequenceItem.addAll(xInput, xOutput);
        };

        DoubleVertex xInitial = new ConstantDoubleVertex(1.0).setLabel(xLabel);
        String sequenceName = "My_Awesome_Sequence";
        VertexDictionary initialState = SimpleVertexDictionary.of(xInitial);

        Sequence sequence = new SequenceBuilder()
            .withInitialState(initialState)
            .named(sequenceName)
            .count(2)
            .withFactory(factory)
            .build();

        Vertex<? extends DoubleTensor> xOutput = sequence.getLastItem().get(xLabel);
        assertThat(xOutput.getValue().scalar(), is(4.0));
        assertThat(xOutput.getLabel().getQualifiedName(), notNullValue());
        assertThat(xOutput.getLabel().getOuterNamespace(), is(Optional.of(sequenceName)));
        assertThat(xOutput.getLabel().getQualifiedName(), startsWith("My_Awesome_Sequence.Sequence_Item_1."));
        assertThat(xOutput.getLabel().getUnqualifiedName(), is("x"));
    }
}
