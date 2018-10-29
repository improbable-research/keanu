package io.improbable.keanu.plating;


import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.plating.loop.Loop;
import io.improbable.keanu.plating.loop.LoopConstructionException;
import io.improbable.keanu.plating.loop.LoopDidNotTerminateException;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexMatchers;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;

public class LoopTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    private final Function<DoubleVertex, DoubleVertex> increment = (v) -> v.plus(1.);
    private final Supplier<BoolVertex> flip = () -> new BernoulliVertex(0.5);
    private final Supplier<BoolVertex> alwaysTrue = () -> ConstantVertex.of(true);
    private final Vertex startValue = ConstantVertex.of(0.);

    @Test
    public void youCanGetTheOutputVertex() {
        Loop loop = Loop
            .withInitialConditions(startValue)
            .iterateWhile(flip)
            .apply(increment);
        Vertex<?> output = loop.getOutput();
        assertThat(output, instanceOf(DoubleVertex.class));
    }

    @Test
    public void thereIsADefaultMaxLength(){
        Loop loop = Loop
            .withInitialConditions(startValue)
            .iterateWhile(alwaysTrue)
            .apply(increment);
        assertThat(loop.getPlates().size(), equalTo(Loop.DEFAULT_MAX_COUNT));
    }

    @Test
    public void itThrowsIfYouGetTheOutputButTheMaxNumberOfIterationsHasBeenReached() {
        expectedException.expect(LoopDidNotTerminateException.class);
        expectedException.expectMessage("Loop has exceeded its max count");
        Loop loop = Loop
            .withInitialConditions(startValue)
            .iterateWhile(alwaysTrue)
            .apply(increment);
        loop.getOutput();
    }

    @Test
    public void youCanTellItNotToThrowWhenTheMaxNumberOfIterationsHaveBeenReached() {
        Loop loop = Loop
            .withInitialConditions(startValue)
            .doNotThrowWhenMaxCountIsReached()
            .iterateWhile(alwaysTrue)
            .apply(increment);
        loop.getOutput();
        // does not throw
    }

    @Test
    public void youCanOverrideTheDefaultMaxLength(){
        int customMaxCount = 5;
        Loop loop = Loop
            .withInitialConditions(startValue)
            .withMaxIterations(customMaxCount)
            .iterateWhile(alwaysTrue)
            .apply(increment);
        assertThat(loop.getPlates().size(), equalTo(customMaxCount));
    }

    @Test
    public void itThrowsIfYouPassInMultipleOutputVertices(){
        expectedException.expect(LoopConstructionException.class);
        expectedException.expectMessage("Duplicate label found in base case");
        Loop.withInitialConditions(ConstantVertex.of(0.).setLabel(Loop.VALUE_OUT_LABEL), ConstantVertex.of(1.).setLabel(Loop.VALUE_OUT_LABEL))
            .iterateWhile(alwaysTrue)
            .apply(increment);
    }

    @Test
    public void youCanLoopUntilAConditionIsTrue() {
        Loop loop = Loop
            .withInitialConditions(startValue)
            .iterateWhile(flip)
            .apply(increment);

        DoubleVertex output = loop.getOutput();

        for (int firstFailure : new int[]{0, 1, 2, 10, Loop.DEFAULT_MAX_COUNT - 1}) {
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
    public void youCanChainTwoLoopsTogetherInABayesNet() {
        Loop loop = Loop
            .withInitialConditions(startValue)
            .doNotThrowWhenMaxCountIsReached()
            .iterateWhile(alwaysTrue)
            .apply(increment);

        Vertex<?> outputFromFirstLoop = loop.getOutput();

        Loop loop2 = Loop
            .withInitialConditions((Vertex<?>) loop.getOutput())
            .doNotThrowWhenMaxCountIsReached()
            .iterateWhile(alwaysTrue)
            .apply(increment);

        Vertex<?> output = loop2.getOutput();

        new BayesianNetwork(output.getConnectedGraph());
    }

    @Test
    public void theConditionCanBeAFunctionOfThePlateVariables() {
        Function<Plate, BoolVertex> lessThanTen = plate -> {
            DoubleVertex valueIn = plate.get(Loop.VALUE_IN_LABEL);
            return valueIn.lessThan(ConstantVertex.of(10.));
        };

        Loop loop = Loop
            .withInitialConditions(startValue)
            .iterateWhile(lessThanTen)
            .apply(increment);

        DoubleVertex output = loop.getOutput();
        assertThat(output, VertexMatchers.hasValue(10.));

    }

    @Test
    public void youCanAddCustomProxyVariableMappings() {
        VertexLabel factorInLabel = new VertexLabel("factorIn");
        VertexLabel factorOutLabel = new VertexLabel("factorOut");
        DoubleVertex startFactorial = ConstantVertex.of(1.);
        DoubleVertex startFactor = ConstantVertex.of(1.).setLabel(factorOutLabel);

        BiFunction<Plate, DoubleVertex, DoubleVertex> factorial = (plate, valueIn) -> {
            DoubleVertex factorIn = new DoubleProxyVertex(factorInLabel);
            DoubleVertex factorOut = factorIn.plus(ConstantVertex.of(1.));
            plate.add(factorIn);
            plate.add(factorOutLabel, factorOut);
            return valueIn.times(factorOut);
        };

        Loop loop = Loop
            .withInitialConditions(startFactorial, startFactor)
            .withMaxIterations(5)
            .doNotThrowWhenMaxCountIsReached()
            .mapping(factorInLabel, factorOutLabel)
            .iterateWhile(alwaysTrue)
            .apply(factorial);

        DoubleVertex output = loop.getOutput();
        assertThat(output, VertexMatchers.hasValue(720.));
    }


}
