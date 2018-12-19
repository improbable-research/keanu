package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class PlaceHolderBoolVertex extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    public PlaceHolderBoolVertex(@LoadShape long[] initialShape) {
        super(initialShape);
    }

    @Override
    public BooleanTensor calculate() {
        return this.getValue();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return this.getValue();
    }
}
