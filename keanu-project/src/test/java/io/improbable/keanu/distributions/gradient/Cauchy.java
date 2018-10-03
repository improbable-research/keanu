package io.improbable.keanu.distributions.gradient;

public class Cauchy {

    private Cauchy() {}

    public static Cauchy.Diff dlnPdf(double location, double scale, double x) {
        final double xMinusLocation = x - location;
        final double xMinusLocationPow2 = xMinusLocation * xMinusLocation;
        final double scalePow2 = scale * scale;
        final double locationTimesXTimes2 = location * x * 2;

        final double dlnP_dlocation = xMinusLocation * 2 / (scalePow2 + xMinusLocationPow2);
        final double dlnP_dscale =
                (xMinusLocationPow2 - scalePow2) / (scale * (xMinusLocationPow2 + scalePow2));
        final double dlnP_dx =
                xMinusLocation
                        * -2
                        / ((location * location) - locationTimesXTimes2 + scalePow2 + (x * x));

        return new Cauchy.Diff(dlnP_dlocation, dlnP_dscale, dlnP_dx);
    }

    public static class Diff {
        public final double dPdlocation;
        public final double dPdscale;
        public final double dPdx;

        public Diff(double dPdlocation, double dPdscale, double dPdx) {
            this.dPdlocation = dPdlocation;
            this.dPdscale = dPdscale;
            this.dPdx = dPdx;
        }
    }
}
