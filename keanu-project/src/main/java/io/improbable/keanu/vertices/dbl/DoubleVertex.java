package io.improbable.keanu.vertices.dbl;


import com.google.common.primitives.Doubles;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.network.NetworkWriter;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MaxVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogGammaVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixDeterminantVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixInverseVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ReshapeVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SliceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

public abstract class DoubleVertex extends Vertex<DoubleTensor> implements DoubleOperators<DoubleVertex>, Differentiable {

    public DoubleVertex(long[] initialShape) {
        super(initialShape);
    }

    /**
     * @param dimension dimension to concat along. Negative dimension indexing is not supported.
     * @param toConcat  array of things to concat. Must match in all dimensions except for the provided
     *                  dimension
     * @return a vertex that represents the concatenation of the toConcat
     */
    public static DoubleVertex concat(int dimension, DoubleVertex... toConcat) {
        return new ConcatenationVertex(dimension, toConcat);
    }

    public static DoubleVertex min(DoubleVertex a, DoubleVertex b) {
        return new MinVertex(a, b);
    }

    public static DoubleVertex max(DoubleVertex a, DoubleVertex b) {
        return new MaxVertex(a, b);
    }

    public DoubleVertex minus(DoubleVertex that) {
        return new DifferenceVertex(this, that);
    }

    public DoubleVertex plus(DoubleVertex that) {
        return new AdditionVertex(this, that);
    }

    public DoubleVertex multiply(DoubleVertex that) {
        return new MultiplicationVertex(this, that);
    }

    public DoubleVertex matrixMultiply(DoubleVertex that) {
        return new MatrixMultiplicationVertex(this, that);
    }

    public DoubleVertex matrixInverse() {
        return new MatrixInverseVertex(this);
    }

    public DoubleVertex matrixDeterminant() {
        return new MatrixDeterminantVertex(this);
    }

    public DoubleVertex divideBy(DoubleVertex that) {
        return new DivisionVertex(this, that);
    }

    public DoubleVertex pow(DoubleVertex exponent) {
        return new PowerVertex(this, exponent);
    }

    @Override
    public DoubleVertex minus(double that) {
        return minus(new ConstantDoubleVertex(that));
    }

    @Override
    public DoubleVertex reverseMinus(double that) {
        return new ConstantDoubleVertex(that).minus(this);
    }

    @Override
    public DoubleVertex plus(double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    public DoubleVertex multiply(double that) {
        return multiply(new ConstantDoubleVertex(that));
    }

    public DoubleVertex divideBy(double that) {
        return divideBy(new ConstantDoubleVertex(that));
    }

    @Override
    public DoubleVertex pow(double that) {
        return pow(new ConstantDoubleVertex(that));
    }

    public DoubleVertex abs() {
        return new AbsVertex(this);
    }

    public DoubleVertex floor() {
        return new FloorVertex(this);
    }

    public DoubleVertex ceil() {
        return new CeilVertex(this);
    }

    public DoubleVertex round() {
        return new RoundVertex(this);
    }

    @Override
    public DoubleVertex exp() {
        return new ExpVertex(this);
    }

    public DoubleVertex log() {
        return new LogVertex(this);
    }

    public DoubleVertex logGamma() {
        return new LogGammaVertex(this);
    }

    public DoubleVertex sigmoid() {
        return new SigmoidVertex(this);
    }

    @Override
    public DoubleVertex sin() {
        return new SinVertex(this);
    }

    @Override
    public DoubleVertex cos() {
        return new CosVertex(this);
    }

    public DoubleVertex tan() {
        return new TanVertex(this);
    }

    @Override
    public DoubleVertex asin() {
        return new ArcSinVertex(this);
    }

    @Override
    public DoubleVertex acos() {
        return new ArcCosVertex(this);
    }

    public DoubleVertex atan() {
        return new ArcTanVertex(this);
    }

    public DoubleVertex atan2(DoubleVertex that) {
        return new ArcTan2Vertex(this, that);
    }

    /**
     * Sum over all dimensions. This will always result in a scalar.
     *
     * @return a vertex representing the summation result
     */
    public DoubleVertex sum() {
        return new SumVertex(this);
    }

    /**
     * Sum over specified dimensions.
     *
     * @param sumOverDimensions dimensions to sum over. Negative dimension indexing is not supported
     * @return a vertex representing the summation result
     */
    public DoubleVertex sum(int... sumOverDimensions) {
        return new SumVertex(this, sumOverDimensions);
    }

    public DoubleVertex reshape(long... proposedShape) {
        return new ReshapeVertex(this, proposedShape);
    }

    public DoubleVertex lambda(long[] outputShape, Function<DoubleTensor, DoubleTensor> op,
                               Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda,
                               Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(outputShape, this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleVertex lambda(Function<DoubleTensor, DoubleTensor> op,
                               Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda,
                               Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    @Override
    public DoubleVertex times(DoubleVertex that) {
        return multiply(that);
    }

    @Override
    public DoubleVertex div(DoubleVertex that) {
        return divideBy(that);
    }

    @Override
    public DoubleVertex times(double that) {
        return multiply(that);
    }

    @Override
    public DoubleVertex div(double that) {
        return divideBy(that);
    }

    @Override
    public DoubleVertex reverseDiv(double that) {
        return new ConstantDoubleVertex(that).div(this);
    }

    @Override
    public DoubleVertex unaryMinus() {
        return multiply(-1.0);
    }

    public BoolVertex equalTo(DoubleVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    public <T extends Tensor> BoolVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex greaterThan(Vertex<T> rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex greaterThanOrEqualTo(Vertex<T> rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex lessThan(Vertex<T> rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex lessThanOrEqualTo(Vertex<T> rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }

    public DoubleVertex take(long... index) {
        return new TakeVertex(this, index);
    }

    public DoubleVertex slice(int dimension, int index) {
        return new SliceVertex(this, dimension, index);
    }

    public void setValue(double value) {
        super.setValue(DoubleTensor.scalar(value));
    }

    public void setValue(double[] values) {
        super.setValue(DoubleTensor.create(values));
    }

    public void setAndCascade(double value) {
        super.setAndCascade(DoubleTensor.scalar(value));
    }

    public void setAndCascade(double[] values) {
        super.setAndCascade(DoubleTensor.create(values));
    }

    public void observe(double value) {
        super.observe(DoubleTensor.scalar(value));
    }

    public void observe(double[] values) {
        super.observe(DoubleTensor.create(values));
    }

    public double getValue(int... index) {
        return getValue().getValue(index);
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        if (isObserved()) {
            return PartialDerivatives.OF_CONSTANT;
        } else {
            return PartialDerivatives.withRespectToSelf(this.getId(), this.getShape());
        }
    }

    @Override
    public void setValue(KeanuSavedBayesNet.VertexValue valueBuf) {
        if (valueBuf.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.DOUBLEVAL) {
            throw new IllegalArgumentException("Non Double Value specified for Double Vertex");
        } else {
            setValue(DoubleTensor.create(Doubles.toArray(valueBuf.getDoubleVal().getValuesList())));
        }
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        return Collections.singletonMap(
            this,
            PartialDerivatives.withRespectToSelf(this.getId(), this.getShape())
        );
    }

    @Override
    public void saveValue(NetworkWriter protobufWriter) {
        protobufWriter.saveValue(this);
    }
}
