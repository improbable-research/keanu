package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class TensorTriangular {

    public static DoubleTensor sample(int[] shape, DoubleTensor xMin, DoubleTensor xMax, DoubleTensor c, KeanuRandom random) {
        final DoubleTensor p = random.nextDouble(shape);
        final DoubleTensor q = p.unaryMinus().plus(1);
        final DoubleTensor range = xMax.minus(xMin);

        final DoubleTensor conditional = (c.minus(xMin)).div(xMax.minus(xMin));

        final DoubleTensor lessThan = xMin.plus((range.times(c.minus(xMin).times(p))).sqrt());
        final DoubleTensor greaterThan = xMax.minus((range.times(xMax.minus(c).times(q))).sqrt());

        final DoubleTensor lessThanMask = p.getLessThanOrEqualToMask(conditional);
        final DoubleTensor greaterThanMask = p.getGreaterThanMask(conditional);

        return (lessThan.times(lessThanMask).plus(greaterThan.times(greaterThanMask)));
    }

    public static DoubleTensor logPdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor c, DoubleTensor x) {
        final DoubleTensor range = xMax.minus(xMin);

        final DoubleTensor conditionalFirstHalf = x.getGreaterThanMask(xMin);
        final DoubleTensor conditionalSecondHalf = x.getLessThanMask(c);
        final DoubleTensor conditionalAnd = conditionalFirstHalf.times(conditionalSecondHalf);
        final DoubleTensor conditionalResult = range.reciprocal().times(2).times(x.minus(xMin)).div(c.minus(xMin));

        final DoubleTensor elseIfConditionalFirstHalf = x.getGreaterThanMask(c);
        final DoubleTensor elseIfConditionalSecondHalf = x.getLessThanOrEqualToMask(xMax);
        final DoubleTensor elseIfConditionalAnd = elseIfConditionalFirstHalf.times(elseIfConditionalSecondHalf);
        final DoubleTensor elseIfConditionalResult = range.reciprocal().times(2).times(xMax.minus(x)).div(xMax.minus(c));

        return (conditionalResult.times(conditionalAnd).plus(elseIfConditionalResult.times(elseIfConditionalAnd))).log();
    }


}
