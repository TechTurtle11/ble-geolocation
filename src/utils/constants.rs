enum Prior {
    UNIFORM,
    LOCAL
}

enum MapAttribute {
    PROB,
    COV
}

enum MeasurementProcess {
    ALL,
    MEAN,
    QUANTILE,
    MEDIAN
}

pub enum Model {
    GAUSSIAN,
    GAUSSIANKNN,
    GAUSSIANMINMAX,
    KNN,
    WKNN,
    PROPOGATION,
    PROXIMITY
}