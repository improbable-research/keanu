package io.improbable.keanu.plating;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import static io.improbable.keanu.vertices.VertexLabelMatchers.hasUnqualifiedName;
import static io.improbable.keanu.vertices.VertexMatchers.hasLabel;
import static io.improbable.keanu.vertices.VertexMatchers.hasNoLabel;
import static io.improbable.keanu.vertices.VertexMatchers.hasParents;

import java.util.Arrays;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.SimpleVertexDictionary;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexMatchers;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;

public class PlateBuilderTest {

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
    public void buildPlatesFromCount_Size() {
        int n = 100;
        Plates plates = new PlateBuilder()
            .count(n)
            .withFactory(plate -> {
            })
            .build();
        assertEquals(n, plates.size());
    }

    @Test
    public void buildPlatesFromCount_PlateContents() {
        int n = 100;
        VertexLabel vertexName = new VertexLabel("vertexName");
        Plates plates = new PlateBuilder<>()
            .count(n)
            .withFactory((plate) -> plate.add(vertexName, new BernoulliVertex(0.5)))
            .build();
        plates.asList().forEach(plate -> {
            assertNotNull(plate.get(vertexName));
        });
    }

    @Test
    public void buildPlatesFromData_Size() {
        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
            })
            .build();
        assertEquals(ROWS.size(), plates.size());
    }

    @Test
    public void buildPlatesFromData_Contents() {
        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
                assertEquals(0, bean.x);
            })
            .build();
    }

    @Test
    public void ifAVertexIsLabeledThatIsWhatsUsedToReferToItInThePlate() {
        VertexLabel label = new VertexLabel("label");

        Vertex<?> startVertex = ConstantVertex.of(1.).labeledAs(label);

        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(startVertex)
            .withTransitionMapping(ImmutableMap.of(label, label))
            .count(10)
            .withFactory((plate) -> {
                DoubleVertex intermediateVertex = new DoubleProxyVertex(label);
                plate.add(intermediateVertex);
            })
            .build();

        for (Plate plate : plates) {
            Vertex<?> vertex = plate.get(label);
            assertThat(vertex, hasLabel(hasUnqualifiedName(label.getUnqualifiedName())));
            Vertex<?> parent = Iterables.getOnlyElement(vertex.getParents());
            assertThat(parent, hasLabel(hasUnqualifiedName(label.getUnqualifiedName())));
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void youCannotAddTheSameLabelTwiceIntoOnePlate() {
        new PlateBuilder<Integer>()
            .count(10)
            .withFactory((plate) -> {
                VertexLabel label = new VertexLabel("x");
                DoubleVertex vertex1 = ConstantVertex.of(1.);
                DoubleVertex vertex2 = ConstantVertex.of(1.);
                plate.add(label, vertex1);
                plate.add(label, vertex2);
            })
            .build();
    }

    @Test
    public void youCanCreateASetOfPlatesWithACommonParameterFromACount() {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Plates plates = new PlateBuilder<Bean>()
            .count(10)
            .withFactory((plate) -> {
                BoolVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                plate.add(label, flip);
            })
            .build();


        for (Plate plate : plates) {
            Vertex<DoubleTensor> flip = plate.get(label);
            assertThat(flip.getParents(), contains(commonTheta));
        }
    }

    @Test
    public void youCanPutThePlatesIntoABayesNet() {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Plates plates = new PlateBuilder<Bean>()
            .count(10)
            .withFactory((plate) -> {
                BoolVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                plate.add(label, flip);
            })
            .build();

        new BayesianNetwork(commonTheta.getConnectedGraph());
    }

    @Test
    public void youCanPutTheSameVertexIntoMultiplePlates() {
        VertexLabel thetaLabel = new VertexLabel("theta");
        VertexLabel flipLabel = new VertexLabel("flip");
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        new PlateBuilder<Bean>()
            .count(10)
            .withFactory((plate) -> {
                BoolVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                plate.add(thetaLabel, commonTheta);
                plate.add(flipLabel, flip);
            })
            .build();

        new BayesianNetwork(commonTheta.getConnectedGraph());
    }


    @Test
    public void youCanCreateASetOfPlatesWithACommonParameterFromAnIterator() {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
                BoolVertex flip = new BernoulliVertex(commonTheta);
                flip.observe(false);
                plate.add(label, flip);
            })
            .build();


        for (Plate plate : plates) {
            Vertex<DoubleTensor> flip = plate.get(label);
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
    public void youCanCreateATimeSeriesFromPlatesFromACount() {

        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xPreviousLabel = PlateBuilder.proxyFor(xLabel);
        VertexLabel yLabel = new VertexLabel("y");

        Vertex<DoubleTensor> initialX = ConstantVertex.of(1.);
        List<Integer> ys = ImmutableList.of(0, 1, 2, 1, 3, 2);

        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(xLabel, initialX)
            .count(10)
            .withFactory((plate) -> {
                DoubleVertex xPrevious = new DoubleProxyVertex(xPreviousLabel);
                DoubleVertex x = new ExponentialVertex(xPrevious);
                IntegerVertex y = new PoissonVertex(x);
                plate.add(xPrevious);
                plate.add(xLabel, x);
                plate.add(yLabel, y);
            })
            .build();


        Vertex<DoubleTensor> previousX = initialX;

        for (Plate plate : plates) {
            Vertex<DoubleTensor> xPreviousProxy = plate.get(xPreviousLabel);
            Vertex<DoubleTensor> x = plate.get(xLabel);
            Vertex<DoubleTensor> y = plate.get(yLabel);
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
    public void youCanCreateATimeSeriesFromPlatesFromAnIterator() {

        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xPreviousLabel = PlateBuilder.proxyFor(xLabel);
        VertexLabel yLabel = new VertexLabel("y");

        Vertex<DoubleTensor> initialX = ConstantVertex.of(1.);
        List<Integer> ys = ImmutableList.of(0, 1, 2, 1, 3, 2);

        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(xLabel, initialX)
            .fromIterator(ys.iterator())
            .withFactory((plate, observedY) -> {
                DoubleVertex xPreviousProxy = new DoubleProxyVertex(xPreviousLabel);
                DoubleVertex x = new ExponentialVertex(xPreviousProxy);
                IntegerVertex y = new PoissonVertex(x);
                y.observe(observedY);
                plate.add(xPreviousProxy);
                plate.add(xLabel, x);
                plate.add(yLabel, y);
            })
            .build();


        Vertex<DoubleTensor> previousX = initialX;

        for (Plate plate : plates) {
            Vertex<DoubleTensor> xPreviousProxy = plate.get(xPreviousLabel);
            Vertex<DoubleTensor> x = plate.get(xLabel);
            Vertex<DoubleTensor> y = plate.get(yLabel);
            assertThat(xPreviousProxy.getParents(), contains(previousX));
            assertThat(x.getParents(), contains(xPreviousProxy));
            assertThat(y.getParents(), contains(x));
            previousX = x;
        }
    }

    /**
     * Note that this behaviour is wrapped by the Loop class
     * See LoopTest.java for example usage
     */
    @Test
    public void youCanCreateALoopFromPlatesFromACount() {
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
        BoolVertex tru = ConstantVertex.of(true);
        DoubleVertex initialValue = ConstantVertex.of(0.);

        int maximumLoopLength = 100;

        Plates plates = new PlateBuilder<Integer>()
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
            .withFactory((plate) -> {
                // inputs
                DoubleVertex runningTotal = new DoubleProxyVertex(runningTotalLabel);
                BoolVertex stillLooping = new BoolProxyVertex(stillLoopingLabel);
                DoubleVertex valueIn = new DoubleProxyVertex(valueInLabel);
                plate.addAll(ImmutableSet.of(runningTotal, stillLooping, valueIn));

                // intermediate
                DoubleVertex one = ConstantVertex.of(1.);
                BoolVertex condition = new BernoulliVertex(0.5);
                plate.add(oneLabel, one);
                plate.add(conditionLabel, condition);

                // outputs
                DoubleVertex plus = runningTotal.plus(one);
                BoolVertex loopAgain = stillLooping.and(condition);
                DoubleVertex result = If.isTrue(loopAgain).then(plus).orElse(valueIn);
                plate.add(plusLabel, plus);
                plate.add(loopLabel, loopAgain);
                plate.add(valueOutLabel, result);
            })
            .build();


        DoubleVertex previousPlus = initialSum;
        BoolVertex previousLoop = tru;
        DoubleVertex previousValueOut = initialValue;

        for (Plate plate : plates) {
            DoubleVertex runningTotal = plate.get(runningTotalLabel);
            BoolVertex stillLooping = plate.get(stillLoopingLabel);
            DoubleVertex valueIn = plate.get(valueInLabel);

            DoubleVertex one = plate.get(oneLabel);
            BoolVertex condition = plate.get(conditionLabel);

            DoubleVertex plus = plate.get(plusLabel);
            BoolVertex loop = plate.get(loopLabel);
            DoubleVertex valueOut = plate.get(valueOutLabel);

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


        DoubleVertex output = plates.asList().get(maximumLoopLength - 1).get(valueOutLabel);

        for (int firstFailure : new int[]{0, 1, 2, 10, 99}) {
            for (Plate plate : plates) {
                BoolVertex condition = plate.get(conditionLabel);
                condition.setAndCascade(true);
            }
            BoolVertex condition = plates.asList().get(firstFailure).get(conditionLabel);
            condition.setAndCascade(false);
            Double expectedOutput = new Double(firstFailure);
            assertThat(output, VertexMatchers.hasValue(expectedOutput));
        }
    }

    @Test
    public void itThrowsIfTheresAProxyVertexThatItDoesntKnowHowToMap() {
        expectedException.expect(PlateConstructionException.class);
        expectedException.expectMessage(startsWith("Cannot find transition mapping for "));
        VertexLabel realLabel = new VertexLabel("real");
        VertexLabel fakeLabel = new VertexLabel("fake");
        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(SimpleVertexDictionary.of())
            .withTransitionMapping(ImmutableMap.of(realLabel, realLabel))
            .count(10)
            .withFactory((plate) -> {
                plate.add(new DoubleProxyVertex(fakeLabel));
            })
            .build();
    }

    @Test
    public void itThrowsIfTheresAProxyVertexButNoBaseCase() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("You must provide a base case for the Transition Vertices - use withInitialState()");
        VertexLabel realLabel = new VertexLabel("real");
        Plates plates = new PlateBuilder<Integer>()
            .withTransitionMapping(ImmutableMap.of(realLabel, realLabel))
            .count(10)
            .withFactory((plate) -> {
                plate.add(new DoubleProxyVertex(realLabel));
            })
            .build();
    }

    @Test
    public void itThrowsIfTheresAnUnknownLabelInTheProxyMapping() {
        expectedException.expect(PlateConstructionException.class);
        expectedException.expectMessage("Cannot find VertexLabel fake");
        VertexLabel realLabel = new VertexLabel("real");
        VertexLabel fakeLabel = new VertexLabel("fake");
        DoubleVertex initialState = ConstantVertex.of(1.);
        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(realLabel, initialState)
            .withTransitionMapping(ImmutableMap.of(realLabel, fakeLabel))
            .count(10)
            .withFactory((plate) -> {
                plate.add(new DoubleProxyVertex(realLabel));
            })
            .build();
    }
}
