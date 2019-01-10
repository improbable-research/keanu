package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.PlaceHolderVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class PlaceHolderBoolVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor>, PlaceHolderVertex {

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
