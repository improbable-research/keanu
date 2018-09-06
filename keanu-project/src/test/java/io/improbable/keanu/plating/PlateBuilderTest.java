package io.improbable.keanu.plating;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
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
            .withFactory((plate) -> plate.add(new BernoulliVertex(0.5).labelled(vertexName)))
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

    @Test(expected = IllegalArgumentException.class)
    public void youCannotAddTheSameLabelTwiceIntoOnePlate() {
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
    public void youCanCreateASetOfPlatesWithACommonParameterFromACount() {
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
    public void youCanCreateASetOfPlatesWithACommonParameterFromAnIterator() {
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
     */
    @Test
    public void youCanCreateATimeSeriesFromPlatesFromACount() {

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
     */
    @Test
    public void youCanCreateATimeSeriesFromPlatesFromAnIterator() {

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
}
