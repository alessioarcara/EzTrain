from collections.abc import Mapping

from eztrain import Image, Metric, MetricCollection


class CountingMetric:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, *args: object, **kwargs: object) -> None:
        self.count += 1

    def compute(self) -> Mapping[str, float]:
        return {self.name: float(self.count)}


class PlottingMetric(CountingMetric):
    def plot(self) -> Mapping[str, Image]:
        return {f"{self.name}_plot": Image(data=None)}


def test_collection_fans_out_and_merges():
    a, b = CountingMetric("a"), CountingMetric("b")
    collection = MetricCollection([a, b])

    collection.update()
    collection.update()
    assert collection.compute() == {"a": 2.0, "b": 2.0}

    collection.reset()
    assert collection.compute() == {"a": 0.0, "b": 0.0}


def test_collection_plot_only_from_metrics_that_have_it():
    collection = MetricCollection([CountingMetric("a"), PlottingMetric("b")])
    visuals = collection.plot()
    assert set(visuals) == {"b_plot"}
    assert isinstance(visuals["b_plot"], Image)


def test_empty_collection():
    collection = MetricCollection()
    collection.reset()
    collection.update(1, 2, key="value")
    assert collection.compute() == {}
    assert collection.plot() == {}
    assert len(collection) == 0


def test_duck_typed_metric_satisfies_protocol():
    assert isinstance(CountingMetric("x"), Metric)
    assert isinstance(MetricCollection(), Metric)
