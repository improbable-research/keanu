package io.improbable.keanu.backend.keanu.compiled;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NumericalEqualsVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.MultiplexerVertex;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import io.improbable.keanu.vertices.tensor.If;
import io.improbable.keanu.vertices.utility.AssertVertex;
import io.improbable.keanu.vertices.utility.GraphAssertionException;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class KeanuCompiledGraphTest {

    @Test
    public void compilesEmptyGraph() {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();
        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, ?> result = computableGraph.compute(Collections.emptyMap());

        assertTrue(result.isEmpty());
    }

    @Test
    public void compilesAddition() {
        assertBinaryDoubleMatches(DoubleVertex::plus);
    }

    @Test
    public void compilesSubtraction() {
        assertBinaryDoubleMatches(DoubleVertex::minus);
    }

    @Test
    public void compilesMultiplication() {
        assertBinaryDoubleMatches(DoubleVertex::times);
    }

    @Test
    public void compilesDivision() {
        assertBinaryDoubleMatches(DoubleVertex::div);
    }

    @Test
    public void compilesATan2() {
        assertBinaryDoubleMatches(DoubleVertex::atan2);
    }

    @Test
    public void compilesPow() {
        assertBinaryDoubleMatches(DoubleVertex::pow);
    }

    @Test
    public void compilesMatrixMultiply() {
        assertBinaryDoubleMatches(new long[]{2, 3}, new long[]{3, 2}, DoubleVertex::matrixMultiply);
    }

    @Test
    public void compilesTensorMultiply() {
        assertBinaryDoubleMatches(new long[]{2, 3, 4}, new long[]{3, 2, 4}, (a, b) -> a.tensorMultiply(b, new int[]{1}, new int[]{0}));
    }

    @Test
    public void compilesSeveralChainedOpsWithConstant() {
        assertBinaryDoubleMatches((a, b) -> a.plus(b).times(b).div(2).minus(a));
    }

    private void assertBinaryDoubleMatches(BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        assertBinaryDoubleMatches(new long[0], new long[0], op);
    }

    private void assertBinaryDoubleMatches(long[] shapeA, long[] shapeB, BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        GaussianVertex A = new GaussianVertex(shapeA, 0, 1);
        GaussianVertex B = new GaussianVertex(shapeB, 0, 1);
        DoubleVertex C = op.apply(A, B);

        assertCompiledIsSameAsVertexEvaluation(A, B, C);
    }

    @Test
    public void canAddDirectlyToGraph() {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        DoubleVertex C = A.times(B);

        compiler.convert(C.getConnectedGraph(), ImmutableList.of(C));

        VariableReference summation = compiler.add(A.getReference(), C.getReference());
        compiler.registerOutput(summation);

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs);

        assertEquals(C.getValue(), result.get(C.getReference()));
        assertEquals(C.getValue().plus(A.getValue()), result.get(summation));
    }

    @Test
    public void canReshapeDouble() {
        assertUnaryDoubleMatches(new long[]{3, 4}, (a) -> a.reshape(6, 2));
    }

    @Test
    public void canSliceDouble() {
        assertUnaryDoubleMatches(new long[]{3, 4}, (a) -> a.slice(1, 2));
    }

    @Test
    public void canConcatDouble() {
        assertBinaryDoubleMatches(new long[]{3, 4}, new long[]{3, 4}, (a, b) -> DoubleVertex.concat(1, a, b));
    }

    @Test
    public void canTakeDouble() {
        assertUnaryDoubleMatches(new long[]{3, 4}, (a) -> a.take(1, 2));
    }

    @Test
    public void compilesSum() {
        assertUnaryDoubleMatches(new long[]{2, 2}, DoubleVertex::sum);
        assertUnaryDoubleMatches(new long[]{2, 2}, (a) -> a.sum(0));
        assertUnaryDoubleMatches(new long[]{2, 2}, (a) -> a.sum(1));
    }

    @Test
    public void compilesSimpleUnaryOps() {
        assertUnaryDoubleMatches(DoubleVertex::abs);
        assertUnaryDoubleMatches(DoubleVertex::cos);
        assertUnaryDoubleMatches(DoubleVertex::acos);
        assertUnaryDoubleMatches(DoubleVertex::exp);
        assertUnaryDoubleMatches(DoubleVertex::log);
        assertUnaryDoubleMatches(DoubleVertex::logGamma);
        assertUnaryDoubleMatches(DoubleVertex::sin);
        assertUnaryDoubleMatches(DoubleVertex::asin);
        assertUnaryDoubleMatches(DoubleVertex::tan);
        assertUnaryDoubleMatches(DoubleVertex::atan);
        assertUnaryDoubleMatches(DoubleVertex::ceil);
        assertUnaryDoubleMatches(DoubleVertex::floor);
        assertUnaryDoubleMatches(DoubleVertex::round);
        assertUnaryDoubleMatches(DoubleVertex::sigmoid);
    }

    @Test
    public void compilesSquareMatrices() {
        assertUnaryDoubleMatches(new long[]{2, 2}, DoubleVertex::matrixDeterminant);
        assertUnaryDoubleMatches(new long[]{2, 2}, DoubleVertex::matrixInverse);
    }

    private void assertUnaryDoubleMatches(Function<DoubleVertex, DoubleVertex> op) {
        assertUnaryDoubleMatches(new long[0], op);
        assertUnaryDoubleMatches(new long[]{2}, op);
        assertUnaryDoubleMatches(new long[]{2, 2}, op);
    }

    private void assertUnaryDoubleMatches(long[] shape, Function<DoubleVertex, DoubleVertex> op) {
        UniformVertex A = new UniformVertex(shape, 0, 1);
        DoubleVertex C = op.apply(A);

        assertCompiledIsSameAsVertexEvaluation(A, C);
    }

    @Test
    public void canCompileDoubleIf() {
        long[] shape = new long[]{4};
        BernoulliVertex predicate = new BernoulliVertex(shape, 0.5);
        GaussianVertex A = new GaussianVertex(shape, 0, 1);
        GaussianVertex B = new GaussianVertex(shape, 0, 1);

        DoubleVertex D = If.isTrue(predicate)
            .then(A)
            .orElse(B);

        assertCompiledIsSameAsVertexEvaluation(A, B, predicate, D);
    }

    @Test
    public void canReshapeInteger() {
        assertUnaryIntegerMatches(new long[]{3, 4}, (a) -> a.reshape(6, 2));
    }

    @Test
    public void canSliceInteger() {
        assertUnaryIntegerMatches(new long[]{3, 4}, (a) -> a.slice(1, 2));
    }

    @Test
    public void canConcatInteger() {
        assertBinaryIntegerMatches(new long[]{3, 4}, new long[]{3, 4}, (a, b) -> IntegerVertex.concat(1, a, b));
    }

    @Test
    public void canTakeInteger() {
        assertUnaryIntegerMatches(new long[]{3, 4}, (a) -> a.take(1, 2));
    }

    @Test
    public void compilesSimpleUnaryIntegerOps() {
        assertUnaryIntegerMatches(IntegerVertex::abs);
    }

    private void assertUnaryIntegerMatches(Function<IntegerVertex, IntegerVertex> op) {
        assertUnaryIntegerMatches(new long[0], op);
    }

    private void assertUnaryIntegerMatches(long[] shape, Function<IntegerVertex, IntegerVertex> op) {
        UniformIntVertex A = new UniformIntVertex(shape, 0, 1);
        IntegerVertex C = op.apply(A);

        assertCompiledIsSameAsVertexEvaluation(A, C);
    }

    private void assertBinaryIntegerMatches(long[] shapeA, long[] shapeB, BiFunction<IntegerVertex, IntegerVertex, IntegerVertex> op) {
        UniformIntVertex A = new UniformIntVertex(shapeA, 0, 1);
        UniformIntVertex B = new UniformIntVertex(shapeB, 0, 1);
        IntegerVertex C = op.apply(A, B);

        assertCompiledIsSameAsVertexEvaluation(A, B, C);
    }

    @Test
    public void canCompileIntegerBroadcast() {
        assertUnaryIntegerMatches(new long[]{2}, v -> v.broadcast(new long[]{2, 2}));
    }

    @Test
    public void canCompileIntegerBooleanIndex() {
        UniformIntVertex A = new UniformIntVertex(new long[]{4}, 0, 1);
        BooleanVertex indices = new ConstantBooleanVertex(new boolean[]{true, true, false, false});
        IntegerVertex result = A.get(indices);

        assertCompiledIsSameAsVertexEvaluation(A, indices, result);
    }

    @Test
    public void canCompileIntegerIf() {
        long[] shape = new long[]{4};
        BernoulliVertex predicate = new BernoulliVertex(shape, 0.5);
        UniformIntVertex A = new UniformIntVertex(shape, 0, 1);
        UniformIntVertex B = new UniformIntVertex(shape, 0, 1);

        IntegerVertex D = If.isTrue(predicate)
            .then(A)
            .orElse(B);

        assertCompiledIsSameAsVertexEvaluation(A, B, predicate, D);
    }

    @Test
    public void canReshapeBoolean() {
        assertUnaryBooleanMatches(new long[]{3, 4}, (a) -> a.reshape(6, 2));
    }

    @Test
    public void canSliceBoolean() {
        assertUnaryBooleanMatches(new long[]{3, 4}, (a) -> a.slice(1, 2));
    }

    @Test
    public void canConcatBoolean() {
        assertBinaryBooleanMatches(new long[]{3, 4}, new long[]{3, 4}, (a, b) -> BooleanVertex.concat(1, a, b));
    }

    @Test
    public void canTakeBoolean() {
        assertUnaryBooleanMatches(new long[]{3, 4}, (a) -> a.take(1, 2));
    }

    @Test
    public void compilesEqualTo() {
        assertBinaryBooleanMatches(BooleanVertex::elementwiseEquals);
    }

    @Test
    public void compilesNotEqualTo() {
        assertBinaryBooleanMatches(BooleanVertex::notEqualTo);
    }

    @Test
    public void compilesAnd() {
        assertBinaryBooleanMatches(BooleanVertex::and);
    }

    @Test
    public void compilesOr() {
        assertBinaryBooleanMatches(BooleanVertex::or);
    }

    @Test
    public void compilesNot() {
        assertUnaryBooleanMatches(new long[]{3, 4}, (a) -> BooleanVertex.not(a));
    }

    @Test
    public void canCompileBooleanIf() {
        long[] shape = new long[]{4};
        BernoulliVertex predicate = new BernoulliVertex(shape, 0.5);
        BernoulliVertex A = new BernoulliVertex(shape, 0.5);
        BernoulliVertex B = new BernoulliVertex(shape, 0.5);

        BooleanVertex D = If.isTrue(predicate)
            .then(A)
            .orElse(B);

        assertCompiledIsSameAsVertexEvaluation(A, B, predicate, D);
    }

    private void assertUnaryBooleanMatches(long[] shape, Function<BooleanVertex, BooleanVertex> op) {
        BernoulliVertex A = new BernoulliVertex(shape, 0.5);
        BooleanVertex C = op.apply(A);

        assertCompiledIsSameAsVertexEvaluation(A, C);
    }

    private void assertBinaryBooleanMatches(BiFunction<BooleanVertex, BooleanVertex, BooleanVertex> op) {
        assertBinaryBooleanMatches(new long[0], new long[0], op);
    }

    private void assertBinaryBooleanMatches(long[] shapeA, long[] shapeB, BiFunction<BooleanVertex, BooleanVertex, BooleanVertex> op) {
        BernoulliVertex A = new BernoulliVertex(shapeA, 0.5);
        BernoulliVertex B = new BernoulliVertex(shapeB, 0.5);

        BooleanVertex C = op.apply(A, B);

        assertCompiledIsSameAsVertexEvaluation(A, B, C);
    }

    @Test
    public void canCompareDoublesWithEpsilon() {
        long[] shape = new long[]{10, 10};
        GaussianVertex A = new GaussianVertex(shape, 0, 1);
        GaussianVertex B = new GaussianVertex(shape, 0, 1);

        assertCompiledIsSameAsVertexEvaluation(A, B, new NumericalEqualsVertex<>(A, B, 0.5));
    }

    @Test
    public void canCompareDoublesAndIntegerWithEpsilon() {
        long[] shape = new long[]{10, 10};
        GaussianVertex A = new GaussianVertex(shape, 0, 1);
        PoissonVertex B = new PoissonVertex(shape, 1);

        assertCompiledIsSameAsVertexEvaluation(A, B, new NumericalEqualsVertex<>(A, B.toDouble(), 1.0));
    }

    @Test
    public void canCompileDoubleCompare() {
        long[] shape = new long[]{10, 10};
        assertDoubleCompareMatches(shape, shape, DoubleVertex::greaterThan);
        assertDoubleCompareMatches(shape, shape, DoubleVertex::greaterThanOrEqual);
        assertDoubleCompareMatches(shape, shape, DoubleVertex::lessThan);
        assertDoubleCompareMatches(shape, shape, DoubleVertex::lessThanOrEqual);
    }

    private void assertDoubleCompareMatches(long[] shapeA, long[] shapeB, BiFunction<DoubleVertex, DoubleVertex, BooleanVertex> op) {

        GaussianVertex A = new GaussianVertex(shapeA, 0, 1);
        GaussianVertex B = new GaussianVertex(shapeB, 0, 1);
        BooleanVertex C = op.apply(A, B);

        assertCompiledIsSameAsVertexEvaluation(A, B, C);
    }

    private enum TestEnum {
        A, B, C, D
    }

    @Test
    public void canCompileGenericTake() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.1, 0.5, 0.8, 0.2));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.9, 0.5, 0.2, 0.8));

        CategoricalVertex<TestEnum> A = new CategoricalVertex<>(selectableValues);
        GenericTensorVertex<TestEnum> C = A.take(1);

        assertCompiledIsSameAsVertexEvaluation(A, C);
    }

    @Test
    public void canCompileGenericSlice() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.1, 0.5, 0.8, 0.2));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.9, 0.5, 0.2, 0.8));

        CategoricalVertex<TestEnum> A = new CategoricalVertex<>(selectableValues);
        GenericTensorVertex<TestEnum> C = A.slice(0, 1);

        assertCompiledIsSameAsVertexEvaluation(A, C);
    }

    @Test
    public void canCompileGenericIf() {

        Map<TestEnum, DoubleVertex> selectableValues = new LinkedHashMap<>();
        selectableValues.put(TestEnum.A, ConstantVertex.of(0.1, 0.5, 0.8, 0.2));
        selectableValues.put(TestEnum.B, ConstantVertex.of(0.9, 0.5, 0.2, 0.8));

        BernoulliVertex predicate = new BernoulliVertex(new long[]{4}, 0.5);
        CategoricalVertex<TestEnum> A = new CategoricalVertex<>(selectableValues);
        CategoricalVertex<TestEnum> B = new CategoricalVertex<>(selectableValues);

        GenericTensorVertex<TestEnum> D = If.isTrue(predicate)
            .then(A)
            .orElse(B);

        assertCompiledIsSameAsVertexEvaluation(A, B, predicate, D);
    }

    @Test
    public void canCompilePrint() {

        DoubleVertex A = new GaussianVertex(0, 1);
        A.print("\"hello'\' \\world\t\b\f\r\n", false);

        assertCompiledIsSameAsVertexEvaluation(A, A.getChildren().iterator().next());
    }

    @Test(expected = GraphAssertionException.class)
    public void canCompileAssertVertex() {
        DoubleVertex A = new HalfGaussianVertex(new long[]{2, 2}, 1);
        AssertVertex assertVertex = A.lessThan(ConstantVertex.of(0.0)).assertTrue();
        assertCompiledIsSameAsVertexEvaluation(A, assertVertex);
    }

    @Test(expected = GraphAssertionException.class)
    public void canCompileAssertVertexWithMessage() {
        DoubleVertex A = new HalfGaussianVertex(new long[]{2, 2}, 1);
        AssertVertex assertVertex = A.lessThan(ConstantVertex.of(0.0)).assertTrue("test error message\n");
        assertCompiledIsSameAsVertexEvaluation(A, assertVertex);
    }

    @Test(expected = GraphAssertionException.class)
    public void canCompileAssertVertexWithLabel() {
        DoubleVertex A = new HalfGaussianVertex(new long[]{2, 2}, 1);
        BooleanVertex assertVertex = A.lessThan(ConstantVertex.of(0.0)).assertTrue().setLabel("test label\n");
        assertCompiledIsSameAsVertexEvaluation(A, assertVertex);
    }

    @Test
    public void conCompiledMultiplexer() {
        IntegerVertex select = new UniformIntVertex(0, 2);

        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleVertex B = new GaussianVertex(0, 1);

        MultiplexerVertex<DoubleTensor> mux = new MultiplexerVertex<>(select, A, B);

        assertCompiledIsSameAsVertexEvaluation(A, B, select, mux);
    }

    private void assertCompiledIsSameAsVertexEvaluation(Vertex<?, ?> A, Vertex<?, ?> B, Vertex<?, ?> C, Vertex<?, ?> D) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();
        compiler.convert(D.getConnectedGraph(), ImmutableList.of(D));

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());
        inputs.put(C.getReference(), C.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs);

        assertEquals(D.getValue(), result.get(D.getReference()));
    }

    private void assertCompiledIsSameAsVertexEvaluation(Vertex<?, ?> A, Vertex<?, ?> B, Vertex<?, ?> C) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();
        compiler.convert(C.getConnectedGraph(), ImmutableList.of(C));

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs);

        assertEquals(C.getValue(), result.get(C.getReference()));
    }

    private void assertCompiledIsSameAsVertexEvaluation(Vertex<?, ?> A, Vertex<?, ?> B) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();
        compiler.convert(B.getConnectedGraph(), ImmutableList.of(B));

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs);

        assertEquals(B.getValue(), result.get(B.getReference()));
    }

}
