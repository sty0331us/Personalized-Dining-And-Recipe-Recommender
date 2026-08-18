from pdr.data.schemas import parse_image_urls, price_to_symbols
from pdr.json_utils import parse_json_object
from pdr.rag.fusion import fuse_hits, minmax, reciprocal_rank_fusion
from pdr.data.schemas import RetrievalHit


def test_price_symbols():
    assert price_to_symbols(4) == "$$$$"
    assert price_to_symbols("$$") == "$$"


def test_parse_image_urls_from_string_list():
    raw = "['https://example.com/a.png','https://example.com/b.png']"
    assert parse_image_urls(raw) == ["https://example.com/a.png", "https://example.com/b.png"]
    assert parse_image_urls("[]") == []


def test_parse_json_object_strips_fences():
    raw = '```json\n{"hello": "world"}\n```'
    assert parse_json_object(raw) == {"hello": "world"}


def test_minmax_constant_array():
    values = minmax([0.4, 0.4, 0.4])
    assert list(values) == [1.0, 1.0, 1.0]


def _hit(hit_id: str, name: str, **scores) -> RetrievalHit:
    return RetrievalHit(id=hit_id, name=name, modality="article", **scores)


def test_entity_aware_fusion_merges_modalities():
    text = [_hit("1000003", "Iron & Embers", text_score=0.9)]
    image = [_hit("1000003", "Iron & Embers", img_score=0.8)]
    pref = [_hit("1000003", "Iron & Embers", pref_score=0.7)]
    fused = fuse_hits(text, image, pref, w_text=0.55, w_img=0.30, w_pref=0.15, top_n=5)
    assert len(fused) == 1
    assert fused[0].name == "Iron & Embers"
    assert fused[0].fused == 1.0


def test_reciprocal_rank_fusion_prefers_consensus():
    a = [_hit("a", "A"), _hit("b", "B")]
    b = [_hit("b", "B"), _hit("a", "A")]
    fused = reciprocal_rank_fusion([a, b], top_n=2)
    assert {row.id for row in fused} == {"a", "b"}
