package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Triangular {

    public static DoubleTensor sample(int[] shape, DoubleTensor xMin, DoubleTensor xMax, DoubleTensor c, KeanuRandom random) {
        final DoubleTensor p = random.nextDouble(shape);
        final DoubleTensor q = p.unaryMinus().plusInPlace(1);
        final DoubleTensor range = xMax.minus(xMin);

        final DoubleTensor conditional = (c.minus(xMin)).divInPlace(xMax.minus(xMin));

        final DoubleTensor lessThan = xMin.plus((range.times(c.minus(xMin).timesInPlace(p))).sqrtInPlace());
        final DoubleTensor greaterThan = xMax.minus((range.timesInPlace(xMax.minus(c).timesInPlace(q))).sqrtInPlace());

        final DoubleTensor lessThanMask = p.getLessThanOrEqualToMask(conditional);
        final DoubleTensor greaterThanMask = p.getGreaterThanMask(conditional);

        return (lessThan.timesInPlace(lessThanMask).plusInPlace(greaterThan.timesInPlace(greaterThanMask)));
    }

    public static DoubleTensor logPdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor c, DoubleTensor x) {
        final DoubleTensor range = xMax.minus(xMin);

        final DoubleTensor conditionalFirstHalf = x.getGreaterThanMask(xMin);
        final DoubleTensor conditionalSecondHalf = x.getLessThanMask(c);
        final DoubleTensor conditionalAnd = conditionalFirstHalf.timesInPlace(conditionalSecondHalf);
        final DoubleTensor conditionalResult = range.reciprocal().timesInPlace(2).timesInPlace(x.minus(xMin)).divInPlace(c.minus(xMin));

        final DoubleTensor elseIfConditionalFirstHalf = x.getGreaterThanMask(c);
        final DoubleTensor elseIfConditionalSecondHalf = x.getLessThanOrEqualToMask(xMax);
        final DoubleTensor elseIfConditionalAnd = elseIfConditionalFirstHalf.timesInPlace(elseIfConditionalSecondHalf);
        final DoubleTensor elseIfConditionalResult = range.reciprocalInPlace().timesInPlace(2).timesInPlace(xMax.minus(x)).divInPlace(xMax.minus(c));

        return (conditionalResult.timesInPlace(conditionalAnd).plusInPlace(elseIfConditionalResult.timesInPlace(elseIfConditionalAnd))).logInPlace();
    }


}
