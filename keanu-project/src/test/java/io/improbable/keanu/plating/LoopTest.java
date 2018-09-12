package io.improbable.keanu.plating;


import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import io.improbable.keanu.plating.loop.Loop;
import io.improbable.keanu.plating.loop.LoopException;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexLabelException;
import io.improbable.keanu.vertices.VertexMatchers;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;

public class LoopTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    private final Function<DoubleVertex, DoubleVertex> increment = (v) -> v.plus(1.);
    private final Supplier<BoolVertex> flip = () -> new BernoulliVertex(0.5);
    private final Supplier<BoolVertex> alwaysTrue = () -> ConstantVertex.of(true);
    private final Vertex startValue = ConstantVertex.of(0.);

    @Test
    public void youCanGetTheOutputVertex() throws VertexLabelException, LoopException {
        Loop loop = Loop
            .startingFrom(startValue)
            .whilst(flip)
            .apply(increment);
        Vertex<? extends Tensor<?>> output = loop.getOutput();
        assertThat(output, instanceOf(DoubleVertex.class));
    }

    @Test
    public void thereIsADefaultMaxLength() throws VertexLabelException {
        Loop loop = Loop
            .startingFrom(startValue)
            .whilst(alwaysTrue)
            .apply(increment);
        assertThat(loop.getPlates().size(), equalTo(Loop.DEFAULT_MAX_COUNT));
    }

    @Test
    public void itThrowsIfYouGetTheOutputButTheMaxNumberOfIterationsHasBeenReached() throws VertexLabelException, LoopException {
        expectedException.expect(LoopException.class);
        expectedException.expectMessage("Loop has exceeded its max count");
        Loop loop = Loop
            .startingFrom(startValue)
            .whilst(alwaysTrue)
            .apply(increment);
        loop.getOutput();
    }

    @Test
    public void youCanTellItNotToThrowWhenTheMaxNumberOfIterationsHaveBeenReached() throws VertexLabelException, LoopException {
        Loop loop = Loop
            .startingFrom(startValue)
            .dontThrowWhenMaxCountIsReached()
            .whilst(alwaysTrue)
            .apply(increment);
        loop.getOutput();
        // does not throw
    }

    @Test
    public void youCanOverrideTheDefaultMaxLength() throws VertexLabelException {
        int customMaxCount = 5;
        Loop loop = Loop
            .startingFrom(startValue)
            .atMost(customMaxCount)
            .whilst(alwaysTrue)
            .apply(increment);
        assertThat(loop.getPlates().size(), equalTo(customMaxCount));
    }

    @Test
    public void itThrowsIfYouPassInMultipleBaseCaseVertexesAndDontLabelTheOutput() throws VertexLabelException {
        expectedException.expect(VertexLabelException.class);
        expectedException.expectMessage("You must pass in a base case, i.e. a vertex labelled with Loop.VALUE_OUT_LABEL");
        Loop.startingFrom(ConstantVertex.of(0.), ConstantVertex.of(1.))
            .whilst(alwaysTrue)
            .apply(increment);
    }

    @Test
    public void youCanLoopUntilAConditionIsTrue() throws VertexLabelException, LoopException {
        Loop loop = Loop
            .startingFrom(startValue)
            .whilst(flip)
            .apply(increment);

        DoubleVertex output = loop.getOutput();

        for (int firstFailure : new int[]{0, 1, 2, 10, Loop.DEFAULT_MAX_COUNT - 1}) {
            System.out.format("Testing loop that fails after %d steps%n", firstFailure);
            for (Plate plate : loop.getPlates()) {
                BoolVertex condition = plate.get(Loop.CONDITION_LABEL);
                condition.setAndCascade(true);
            }
            BoolVertex condition = loop.getPlates().asList().get(firstFailure).get(Loop.CONDITION_LABEL);
            condition.setAndCascade(false);
            Double expectedOutput = new Double(firstFailure);
            assertThat(output, VertexMatchers.hasValue(expectedOutput));
        }
    }

    @Test
    public void theConditionCanBeAFunctionOfThePlateVariables() throws VertexLabelException, LoopException {
        Function<Plate, BoolVertex> lessThanTen = plate -> {
            DoubleVertex valueIn = plate.get(Loop.VALUE_IN_LABEL);
            return valueIn.lessThan(ConstantVertex.of(10.));
        };

        Loop loop = Loop
            .startingFrom(startValue)
            .whilst(lessThanTen)
            .apply(increment);

        DoubleVertex output = loop.getOutput();
        assertThat(output, VertexMatchers.hasValue(10.));

    }

    @Test
    public void youCanAddCustomProxyVariableMappings() throws VertexLabelException, LoopException {
        VertexLabel factorInLabel = new VertexLabel("factorIn");
        VertexLabel factorOutLabel = new VertexLabel("factorOut");
        DoubleVertex startFactorial = ConstantVertex.of(1.).labelled(Loop.VALUE_OUT_LABEL);
        DoubleVertex startFactor = ConstantVertex.of(1.).labelled(factorOutLabel);

        BiFunction<Plate, DoubleVertex, DoubleVertex> factorial = (plate, valueIn) -> {
            DoubleVertex factorIn = new DoubleProxyVertex(factorInLabel);
            DoubleVertex factorOut = factorIn.plus(ConstantVertex.of(1.)).labelled(factorOutLabel);
            plate.add(factorIn);
            plate.add(factorOut);
            return valueIn.times(factorOut);
        };

        Loop loop = Loop
            .startingFrom(startFactorial, startFactor)
            .atMost(5)
            .dontThrowWhenMaxCountIsReached()
            .mapping(factorInLabel, factorOutLabel)
            .whilst(alwaysTrue)
            .apply(factorial);

        DoubleVertex output = loop.getOutput();
        assertThat(output, VertexMatchers.hasValue(720.));
    }


}
