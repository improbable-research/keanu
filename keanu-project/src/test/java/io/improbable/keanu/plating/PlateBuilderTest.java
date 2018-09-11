package io.improbable.keanu.plating;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.Arrays;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
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
    public void buildPlatesFromCount_Size() throws VertexLabelException {
        int n = 100;
        Plates plates = new PlateBuilder()
            .count(n)
            .withFactory(plate -> {
            })
            .build();
        assertEquals(n, plates.size());
    }

    @Test
    public void buildPlatesFromCount_PlateContents() throws VertexLabelException {
        int n = 100;
        VertexLabel vertexName = new VertexLabel("vertexName");
        Plates plates = new PlateBuilder<>()
            .count(n)
            .withFactory((plate) -> plate.add(new BernoulliVertex(0.5).labelled(vertexName)))
            .build();
        plates.asList().forEach(plate -> {
            assertNotNull(plate.get(vertexName));
        });
    }

    @Test
    public void buildPlatesFromData_Size() throws VertexLabelException {
        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
            })
            .build();
        assertEquals(ROWS.size(), plates.size());
    }

    @Test
    public void buildPlatesFromData_Contents() throws VertexLabelException {
        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
                assertEquals(0, bean.x);
            })
            .build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void youCannotAddTheSameLabelTwiceIntoOnePlate() throws VertexLabelException {
        new PlateBuilder<Integer>()
            .count(10)
            .withFactory((plate) -> {
                VertexLabel label = new VertexLabel("x");
                DoubleVertex vertex1 = ConstantVertex.of(1.).labelled(label);
                DoubleVertex vertex2 = ConstantVertex.of(1.).labelled(label);
                plate.add(vertex1);
                plate.add(vertex2);
            })
            .build();

    }

    @Test
    public void youCanCreateASetOfPlatesWithACommonParameterFromACount() throws VertexLabelException {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Plates plates = new PlateBuilder<Bean>()
            .count(10)
            .withFactory((plate) -> {
                BoolVertex flip = new BernoulliVertex(commonTheta).labelled(label);
                flip.observe(false);
                plate.add(flip);
            })
            .build();


        for (Plate plate : plates) {
            Vertex<DoubleTensor> flip = plate.get(label);
            assertThat(flip.getParents(), contains(commonTheta));
        }
    }

    @Test
    public void youCanPutThePlatesIntoABayesNet() throws VertexLabelException {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Plates plates = new PlateBuilder<Bean>()
            .count(10)
            .withFactory((plate) -> {
                BoolVertex flip = new BernoulliVertex(commonTheta).labelled(label);
                flip.observe(false);
                plate.add(flip);
            })
            .build();

        new BayesianNetwork(commonTheta.getConnectedGraph());
    }


    @Test
    public void youCanCreateASetOfPlatesWithACommonParameterFromAnIterator() throws VertexLabelException {
        GaussianVertex commonTheta = new GaussianVertex(0.5, 0.01);

        VertexLabel label = new VertexLabel("flip");

        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
                BoolVertex flip = new BernoulliVertex(commonTheta).labelled(label);
                flip.observe(false);
                plate.add(flip);
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
    public void youCanCreateATimeSeriesFromPlatesFromACount() throws VertexLabelException {

        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xPreviousLabel = new VertexLabel("xPrevious");
        VertexLabel yLabel = new VertexLabel("y");

        Vertex<DoubleTensor> initialX = ConstantVertex.of(1.).labelled(xLabel);
        List<Integer> ys = ImmutableList.of(0, 1, 2, 1, 3, 2);

        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(initialX)
            .withProxyMapping(ImmutableMap.of(xPreviousLabel, xLabel))
            .count(10)
            .withFactory((plate) -> {
                DoubleVertex xPrevious = new DoubleProxyVertex(xPreviousLabel);
                DoubleVertex x = new ExponentialVertex(xPrevious).labelled(xLabel);
                IntegerVertex y = new PoissonVertex(x).labelled(yLabel);
                plate.add(xPrevious);
                plate.add(x);
                plate.add(y);
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
    public void youCanCreateATimeSeriesFromPlatesFromAnIterator() throws VertexLabelException {

        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel xPreviousLabel = new VertexLabel("xPreviousProxy");
        VertexLabel yLabel = new VertexLabel("y");

        Vertex<DoubleTensor> initialX = ConstantVertex.of(1.).labelled(xLabel);
        List<Integer> ys = ImmutableList.of(0, 1, 2, 1, 3, 2);

        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(initialX)
            .withProxyMapping(ImmutableMap.of(xPreviousLabel, xLabel))
            .fromIterator(ys.iterator())
            .withFactory((plate, observedY) -> {
                DoubleVertex xPreviousProxy = new DoubleProxyVertex(xPreviousLabel);
                DoubleVertex x = new ExponentialVertex(xPreviousProxy).labelled(xLabel);
                IntegerVertex y = new PoissonVertex(x).labelled(yLabel);
                y.observe(observedY);
                plate.add(xPreviousProxy);
                plate.add(x);
                plate.add(y);
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

    @Test
    public void itThrowsIfTheresAProxyVertexThatItDoesntKnowHowToMap() throws VertexLabelException {
        expectedException.expect(VertexLabelException.class);
        expectedException.expectMessage(startsWith("Cannot find proxy mapping for "));
        VertexLabel realLabel = new VertexLabel("real");
        VertexLabel fakeLabel = new VertexLabel("fake");
        Plates plates = new PlateBuilder<Integer>()
            .withInitialState()
            .withProxyMapping(ImmutableMap.of(realLabel, realLabel))
            .count(10)
            .withFactory((plate) -> {
                plate.add(new DoubleProxyVertex(fakeLabel));
            })
            .build();
    }

    @Test
    public void itThrowsIfTheresAProxyVertexButNoBaseCase() throws VertexLabelException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("You must provide a base case for the Proxy Vertices - use withInitialState()");
        VertexLabel realLabel = new VertexLabel("real");
        Plates plates = new PlateBuilder<Integer>()
            .withProxyMapping(ImmutableMap.of(realLabel, realLabel))
            .count(10)
            .withFactory((plate) -> {
                plate.add(new DoubleProxyVertex(realLabel));
            })
            .build();
    }

    @Test
    public void itThrowsIfTheresAnUnknownLabelInTheProxyMapping() throws VertexLabelException {
        expectedException.expect(VertexLabelException.class);
        expectedException.expectMessage("Cannot find VertexLabel fake");
        VertexLabel realLabel = new VertexLabel("real");
        VertexLabel fakeLabel = new VertexLabel("fake");
        Plates plates = new PlateBuilder<Integer>()
            .withInitialState(ConstantVertex.of(1.).labelled(realLabel))
            .withProxyMapping(ImmutableMap.of(realLabel, fakeLabel))
            .count(10)
            .withFactory((plate) -> {
                plate.add(new DoubleProxyVertex(realLabel));
            })
            .build();
    }


}
