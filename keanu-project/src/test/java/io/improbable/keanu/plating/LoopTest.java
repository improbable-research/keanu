package io.improbable.keanu.plating;


import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;

import java.util.function.Function;
import java.util.function.Supplier;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import io.improbable.keanu.plating.loop.Loop;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.VertexMatchers;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class LoopTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    private final Function<DoubleVertex, DoubleVertex> increment = (v) -> v.plus(1.);
    private final Supplier<BoolVertex> flip = () -> new BernoulliVertex(0.5);
    private final Supplier<BoolVertex> alwaysTrue = () -> ConstantVertex.of(true);
    private final Vertex startValue = ConstantVertex.of(0.);

    @Test
    public void youCanGetTheOutputVertex() throws VertexLabelException {
        Loop loop = Loop
            .startingFrom(startValue)
            .apply(increment)
            .whilst(flip);
        Vertex<? extends Tensor<?>> output = loop.getOutput();
        assertThat(output, instanceOf(DoubleVertex.class));
    }

    @Test
    public void thereIsADefaultMaxLength() throws VertexLabelException {
        Loop loop = Loop
            .startingFrom(startValue)
            .apply(increment)
            .whilst(alwaysTrue);
        assertThat(loop.getPlates().size(), equalTo(Loop.DEFAULT_MAX_COUNT));
    }

    @Test
    public void itThrowsIfYouGetTheOutputButTheMaxNumberOfIterationsHasBeenReached() throws VertexLabelException {
        expectedException.expect(PlateException.class);
        expectedException.expectMessage("Loop has exceeded its max count");
        Loop loop = Loop
            .startingFrom(startValue)
            .apply(increment)
            .whilst(alwaysTrue);
        loop.getOutput();
    }

    @Test
    public void youCanTellItNotToThrowWhenTheMaxNumberOfIterationsHaveBeenReached() throws VertexLabelException {
        Loop loop = Loop
            .startingFrom(startValue)
            .dontThrowWhenMaxCountIsReached()
            .apply(increment)
            .whilst(alwaysTrue);
        loop.getOutput();
        // does not throw
    }

    @Test
    public void youCanOverrideTheDefaultMaxLength() throws VertexLabelException {
        int customMaxCount = 5;
        Loop loop = Loop
            .startingFrom(startValue)
            .atMost(customMaxCount)
            .apply(increment)
            .whilst(alwaysTrue);
        assertThat(loop.getPlates().size(), equalTo(customMaxCount));
    }

    @Test
    public void itThrowsIfYouPassInMultipleBaseCaseVertexesAndDontLabelTheOutput() throws VertexLabelException {
        expectedException.expect(PlateException.class);
        expectedException.expectMessage("You must pass in a base case, i.e. a vertex labelled with Loop.VALUE_OUT_LABEL");
        Loop.startingFrom(ConstantVertex.of(0.), ConstantVertex.of(1.))
        .apply(increment)
        .whilst(alwaysTrue);
    }

    @Test
    public void youCanLoopUntilAConditionIsTrue() throws VertexLabelException {
        Loop loop = Loop
            .startingFrom(startValue)
            .apply(increment)
            .whilst(flip);

        DoubleVertex output = loop.getOutput();
        VertexLabel conditionLabel = new VertexLabel("condition");

        for (int firstFailure : new int[] {0, 1, 2, 10, Loop.DEFAULT_MAX_COUNT - 1}) {
            System.out.format("Testing loop that fails after %d steps%n", firstFailure);
            for (Plate plate : loop.getPlates()) {
                BoolVertex condition = plate.get(conditionLabel);
                condition.setAndCascade(true);
            }
            BoolVertex condition = loop.getPlates().asList().get(firstFailure).get(conditionLabel);
            condition.setAndCascade(false);
            Double expectedOutput = new Double(firstFailure);
            assertThat(output, VertexMatchers.hasValue(expectedOutput));
        }
    }


}
