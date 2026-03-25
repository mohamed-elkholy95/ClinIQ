"""Unit tests for the CompositeNERModel voting strategies.

Validates union, intersection, and majority voting when combining
multiple NER model outputs.
"""


from app.ml.ner.model import CompositeNERModel, Entity, RuleBasedNERModel


def _entity(text: str, etype: str, start: int, end: int, conf: float = 0.9) -> Entity:
    """Convenience factory for Entity instances."""
    return Entity(
        text=text,
        entity_type=etype,
        start_char=start,
        end_char=end,
        confidence=conf,
    )


class _StubNER(RuleBasedNERModel):
    """Returns a fixed set of entities regardless of input."""

    def __init__(self, entities: list[Entity]):
        super().__init__(model_name="stub")
        self._fixed = entities

    def load(self) -> None:
        self._is_loaded = True

    def extract_entities(self, text: str) -> list[Entity]:
        return list(self._fixed)


class TestUnionVoting:
    """Union voting returns entities found by *any* model."""

    def test_union_merges_non_overlapping(self):
        m1 = _StubNER([_entity("aspirin", "MEDICATION", 0, 7)])
        m2 = _StubNER([_entity("500mg", "DOSAGE", 8, 13)])
        composite = CompositeNERModel(models=[m1, m2], voting="union")
        composite.load()

        entities = composite.extract_entities("aspirin 500mg")
        texts = {e.text for e in entities}
        assert "aspirin" in texts
        assert "500mg" in texts

    def test_union_deduplicates_identical(self):
        shared = _entity("aspirin", "MEDICATION", 0, 7)
        m1 = _StubNER([shared])
        m2 = _StubNER([shared])
        composite = CompositeNERModel(models=[m1, m2], voting="union")
        composite.load()

        entities = composite.extract_entities("aspirin")
        assert len(entities) == 1


class TestIntersectionVoting:
    """Intersection voting returns only entities found by *all* models."""

    def test_intersection_keeps_common_only(self):
        common = _entity("aspirin", "MEDICATION", 0, 7)
        extra = _entity("500mg", "DOSAGE", 8, 13)
        m1 = _StubNER([common, extra])
        m2 = _StubNER([common])
        composite = CompositeNERModel(models=[m1, m2], voting="intersection")
        composite.load()

        entities = composite.extract_entities("aspirin 500mg")
        texts = {e.text for e in entities}
        assert "aspirin" in texts
        assert "500mg" not in texts

    def test_intersection_empty_when_no_overlap(self):
        m1 = _StubNER([_entity("aspirin", "MEDICATION", 0, 7)])
        m2 = _StubNER([_entity("500mg", "DOSAGE", 8, 13)])
        composite = CompositeNERModel(models=[m1, m2], voting="intersection")
        composite.load()

        entities = composite.extract_entities("aspirin 500mg")
        assert len(entities) == 0


class TestMajorityVoting:
    """Majority voting keeps entities found by >50% of models."""

    def test_majority_keeps_entity_in_two_of_three(self):
        common = _entity("aspirin", "MEDICATION", 0, 7)
        rare = _entity("500mg", "DOSAGE", 8, 13)
        m1 = _StubNER([common, rare])
        m2 = _StubNER([common])
        m3 = _StubNER([common])
        composite = CompositeNERModel(models=[m1, m2, m3], voting="majority")
        composite.load()

        entities = composite.extract_entities("aspirin 500mg")
        texts = {e.text for e in entities}
        assert "aspirin" in texts

    def test_majority_drops_entity_in_one_of_three(self):
        rare = _entity("500mg", "DOSAGE", 8, 13)
        m1 = _StubNER([rare])
        m2 = _StubNER([])
        m3 = _StubNER([])
        composite = CompositeNERModel(models=[m1, m2, m3], voting="majority")
        composite.load()

        entities = composite.extract_entities("aspirin 500mg")
        texts = {e.text for e in entities}
        assert "500mg" not in texts


class TestCompositeEdgeCases:
    """Edge cases for the composite model."""

    def test_empty_models_list(self):
        composite = CompositeNERModel(models=[], voting="union")
        composite.load()
        entities = composite.extract_entities("something")
        assert entities == []

    def test_single_model_behaves_like_passthrough(self):
        m = _StubNER([_entity("aspirin", "MEDICATION", 0, 7)])
        composite = CompositeNERModel(models=[m], voting="union")
        composite.load()
        entities = composite.extract_entities("aspirin")
        assert len(entities) == 1
        assert entities[0].text == "aspirin"

    def test_is_loaded_after_load(self):
        m = _StubNER([])
        composite = CompositeNERModel(models=[m])
        assert not composite.is_loaded
        composite.load()
        assert composite.is_loaded
